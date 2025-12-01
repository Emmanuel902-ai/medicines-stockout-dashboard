import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import logging
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px


# --------------------------------------------------
# LOGGING SETUP
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.info("Starting Essential Medicines Stockout Early Warning dashboard.")


# --------------------------------------------------
# PATHS & DATA LOADING
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Main LMIS-style time series
try:
    df = pd.read_csv(DATA_DIR / "synthetic_lmis_dataset.csv", parse_dates=["week_start"])
    logger.info(
        "Loaded synthetic LMIS dataset from %s with %d rows.",
        DATA_DIR / "synthetic_lmis_dataset.csv",
        len(df),
    )
except Exception as e:
    logger.exception("Failed to load main LMIS dataset.")
    raise RuntimeError(
        "Failed to load main LMIS dataset. "
        "Check that data/synthetic_lmis_dataset.csv exists and is readable."
    ) from e

# Current status / risk table (already enriched in Colab)
try:
    current_status = pd.read_csv(
        DATA_DIR / "current_status_stockout_risk.csv",
        parse_dates=["current_week", "projected_stockout_date"],
    )
    logger.info(
        "Loaded current status / stockout risk table from %s with %d rows.",
        DATA_DIR / "current_status_stockout_risk.csv",
        len(current_status),
    )
except Exception as e:
    logger.exception("Failed to load current status / stockout risk table.")
    raise RuntimeError(
        "Failed to load current status / stockout risk table. "
        "Check that data/current_status_stockout_risk.csv exists and is readable."
    ) from e

# Seasonality (weekly demand pattern)
try:
    seasonality = pd.read_csv(DATA_DIR / "seasonality_by_week_of_year.csv")
    logger.info(
        "Loaded seasonality table from %s with %d rows.",
        DATA_DIR / "seasonality_by_week_of_year.csv",
        len(seasonality),
    )
except Exception as e:
    logger.exception("Failed to load seasonality table.")
    raise RuntimeError(
        "Failed to load seasonality table. "
        "Check that data/seasonality_by_week_of_year.csv exists and is readable."
    ) from e

# Feature order for ML (if you ever want to re-score from raw features)
try:
    with open(MODELS_DIR / "feature_order.json", "r") as f:
        feature_order = json.load(f)
    logger.info("Loaded feature_order.json from %s.", MODELS_DIR)
except Exception as e:
    logger.exception("Failed to load feature_order.json.")
    raise RuntimeError(
        "Failed to load feature_order.json. "
        "Check that models/feature_order.json exists and is readable."
    ) from e

# Calibrated models (not strictly required for the dashboard,
# because risk_4w is already in current_status, but we keep them loaded).
try:
    xgb_cal = joblib.load(MODELS_DIR / "xgb_stockout_calibrated.pkl")
    rf_cal = joblib.load(MODELS_DIR / "rf_stockout_calibrated.pkl")
    logger.info("Loaded calibrated models from %s.", MODELS_DIR)
except Exception as e:
    logger.exception(
        "Failed to load calibrated models (xgb_stockout_calibrated.pkl / rf_stockout_calibrated.pkl). "
        "Dashboard will continue using precomputed risk_4w values."
    )
    # Models are optional for this prototype – keep app running
    xgb_cal = None
    rf_cal = None

# Basic sanity checks
df = df.sort_values(["product_name", "week_start"]).reset_index(drop=True)
current_status = current_status.sort_values("product_name").reset_index(drop=True)
seasonality = seasonality.sort_values(["product_name", "week_of_year"]).reset_index(
    drop=True
)

PRODUCTS = current_status["product_name"].tolist()
logger.info("Initialised dashboard with %d products.", len(PRODUCTS))


# --------------------------------------------------
# HELPER FUNCTIONS – FORECAST, RISK, SEASONALITY
# --------------------------------------------------
def make_forecast_figure(product_name, forecast_horizon=26):
    """
    Simple, robust forecasting view using Holt–Winters.
    Overlays: train, test, forecast (last 26 weeks) + stockouts.
    """
    g = (
        df[df["product_name"] == product_name]
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    if len(g) <= forecast_horizon + 10:
        # Too short series: just show raw history
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=g["week_start"],
                y=g["quantity_dispensed"],
                mode="lines+markers",
                name="Demand (dispensed)",
            )
        )
        fig.update_layout(
            title=f"Weekly Demand – {product_name} (series too short for forecast)",
            xaxis_title="Week",
            yaxis_title="Quantity dispensed",
            hovermode="x unified",
        )
        return fig

    y = g["quantity_dispensed"].astype(float).values
    dates = g["week_start"].values

    train_size = len(y) - forecast_horizon
    y_train = y[:train_size]
    y_test = y[train_size:]
    dates_train = dates[:train_size]
    dates_test = dates[train_size:]

    # Holt–Winters model (weekly data with yearly seasonality)
    hw_model = ExponentialSmoothing(
        y_train, trend="add", seasonal="add", seasonal_periods=52
    ).fit(optimized=True)
    hw_forecast = hw_model.forecast(forecast_horizon)

    fig = go.Figure()

    # Historical train
    fig.add_trace(
        go.Scatter(
            x=dates_train,
            y=y_train,
            mode="lines",
            name="Train (historical)",
        )
    )

    # Actual test
    fig.add_trace(
        go.Scatter(
            x=dates_test,
            y=y_test,
            mode="lines+markers",
            name="Test (actual)",
        )
    )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=dates_test,
            y=hw_forecast,
            mode="lines",
            name="Holt–Winters forecast",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        title=f"Weekly Demand Forecast – {product_name}",
        xaxis_title="Week",
        yaxis_title="Quantity dispensed",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


def make_stock_timeseries_figure(product_name, safety_weeks=2):
    """
    Stock on hand over time + implied safety stock line + stockout markers.
    """
    g = (
        df[df["product_name"] == product_name]
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    # Rolling AMC (8 weeks) to derive safety stock
    g["amc_8w"] = g["quantity_dispensed"].rolling(8).mean()
    g["safety_stock"] = g["amc_8w"] * safety_weeks

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=g["week_start"],
            y=g["closing_stock"],
            mode="lines+markers",
            name="Stock on hand",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=g["week_start"],
            y=g["safety_stock"],
            mode="lines",
            name=f"Safety stock (~{safety_weeks} weeks)",
            line=dict(dash="dash"),
        )
    )

    stockout_points = g[g["stockout_flag"] == 1]
    if not stockout_points.empty:
        fig.add_trace(
            go.Scatter(
                x=stockout_points["week_start"],
                y=stockout_points["closing_stock"],
                mode="markers",
                name="Stockout event",
                marker=dict(symbol="x", size=10, color="red"),
            )
        )

    fig.update_layout(
        title=f"Stock Level Over Time – {product_name}",
        xaxis_title="Week",
        yaxis_title="Stock on hand",
        hovermode="x unified",
    )
    return fig


def make_risk_bar_figure(selected_product=None):
    """
    Bar chart of 4-week stockout risk for all products.
    Colour by risk band; highlight selected product.
    """
    cs = current_status.copy()
    cs = cs.sort_values("risk_4w", ascending=False)

    color_map = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}
    colors = [color_map.get(b, "#6c757d") for b in cs["risk_band"]]

    fig = px.bar(
        cs,
        x="product_name",
        y="risk_4w",
        color="risk_band",
        color_discrete_map=color_map,
        labels={
            "risk_4w": "P(stockout in next 4 weeks)",
            "product_name": "Product",
            "risk_band": "Risk band",
        },
        title="4-week Stockout Risk by Product",
    )

    # Emphasize selected product outline
    if selected_product is not None:
        marker_line_width = [
            3 if p == selected_product else 0 for p in cs["product_name"]
        ]
        fig.update_traces(marker_line_width=marker_line_width, marker_line_color="black")

    fig.update_layout(xaxis_tickangle=-45, hovermode="x")
    return fig


def make_seasonality_figure(product_name):
    """
    Seasonal demand pattern: average quantity_dispensed by week-of-year.
    """
    ps = seasonality[seasonality["product_name"] == product_name]
    if ps.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No seasonality information for {product_name}",
            xaxis_title="Week of year",
            yaxis_title="Average quantity dispensed",
        )
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ps["week_of_year"],
            y=ps["quantity_dispensed"],
            mode="lines+markers",
            name="Avg weekly demand",
        )
    )
    fig.update_layout(
        title=f"Seasonal Demand Pattern – {product_name}",
        xaxis_title="Week of year (1–52)",
        yaxis_title="Average quantity dispensed",
        hovermode="x unified",
    )
    return fig


# --------------------------------------------------
# GLOBAL KPIs (for top of app)
# --------------------------------------------------
total_products = len(current_status)
high_risk = (current_status["risk_band"] == "High").sum()
medium_risk = (current_status["risk_band"] == "Medium").sum()
low_risk = (current_status["risk_band"] == "Low").sum()

immediate_stockout = (current_status["projected_weeks_to_stockout"] <= 4).sum()


# --------------------------------------------------
# APP INIT
# --------------------------------------------------
external_stylesheets = [
    dbc.themes.FLATLY,  # clean professional theme
]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    title="Essential Medicines – Stockout Early Warning",
)

server = app.server  # for production hosting if needed

product_options = [{"label": p, "value": p} for p in PRODUCTS]


# --------------------------------------------------
# GLOBAL HEADER & FOOTER
# --------------------------------------------------
header = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.H2(
                        "Essential Medicines – Stockout Early Warning",
                        className="mb-0 text-white fw-semibold",
                    ),
                    html.Small(
                        "Prototype analytics dashboard for planners and supply chain teams",
                        className="text-white-50",
                    ),
                ],
                className="d-flex flex-column",
            ),
        ],
        fluid=True,
    ),
    color="primary",
    dark=True,
    className="shadow-sm mb-3",
)

footer = html.Footer(
    dbc.Container(
        html.Small(
            "Prototype built on synthetic LMIS data – for demonstration and learning only.",
            className="text-muted",
        ),
        fluid=True,
    ),
    className="py-3 border-top bg-white mt-3",
)


# --------------------------------------------------
# LAYOUT – SIDEBAR + THREE TABS (MULTI-PAGE)
# --------------------------------------------------
sidebar = dbc.Col(
    [
        html.Div(
            [
                html.H5("Essential Medicines Dashboard", className="fw-bold mb-0"),
                html.Small(
                    "Stockout early warning system",
                    className="text-muted",
                ),
            ],
            className="mb-3",
        ),
        html.Hr(className="my-2"),
        dbc.Nav(
            [
                dbc.NavLink("1. Forecasting", href="/forecasting", id="link-forecast"),
                dbc.NavLink(
                    "2. Early Warning",
                    href="/early-warning",
                    id="link-earlywarning",
                ),
                dbc.NavLink(
                    "3. Seasonality",
                    href="/seasonality",
                    id="link-seasonality",
                ),
            ],
            vertical=True,
            pills=True,
            className="nav-links",
        ),
        html.Hr(className="my-3"),
        html.Div(
            [
                html.Small(
                    "This is a research prototype – not for real clinical or procurement decisions.",
                    className="text-muted",
                )
            ],
            className="small-note",
        ),
    ],
    width=3,
    className="bg-white sidebar-column border-end",
)

content = dbc.Col(
    [
        dcc.Location(id="url"),
        html.Div(id="page-content", className="p-3 p-md-4 content-area"),
    ],
    width=9,
)

app.layout = dbc.Container(
    [
        header,
        dbc.Row(
            [
                sidebar,
                content,
            ],
            className="g-0 main-row",
        ),
        footer,
    ],
    fluid=True,
    className="app-container",
)


# --------------------------------------------------
# PAGE LAYOUTS
# --------------------------------------------------
def layout_overview_cards():
    return dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Products monitored"),
                        dbc.CardBody(
                            html.H3(f"{total_products}", className="card-title")
                        ),
                    ]
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("High-risk SKUs (4-week horizon)"),
                        dbc.CardBody(
                            html.H3(
                                f"{high_risk}", className="card-title text-danger"
                            )
                        ),
                    ]
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Medium-risk SKUs"),
                        dbc.CardBody(
                            html.H3(
                                f"{medium_risk}", className="card-title text-warning"
                            )
                        ),
                    ]
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Projected stockout ≤ 4 weeks"),
                        dbc.CardBody(
                            html.H3(
                                f"{immediate_stockout}",
                                className="card-title text-danger",
                            )
                        ),
                    ]
                ),
                md=3,
            ),
        ],
        className="mb-4",
    )


def layout_forecasting_page():
    return html.Div(
        [
            html.H3("1. Forecasting Dashboard – Demand & Stock Levels"),
            html.P(
                "This view compares historical demand with a Holt–Winters forecast "
                "to support procurement planning and scenario discussions.",
                className="text-muted",
            ),
            html.Hr(),
            layout_overview_cards(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select product"),
                            dcc.Dropdown(
                                id="forecast-product-dropdown",
                                options=product_options,
                                value=PRODUCTS[0],
                                clearable=False,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id="forecast-demand-figure"),
                        md=7,
                    ),
                    dbc.Col(
                        dcc.Graph(id="forecast-stock-figure"),
                        md=5,
                    ),
                ]
            ),
        ]
    )


def layout_early_warning_page():
    return html.Div(
        [
            html.H3("2. Early-Warning Dashboard – 4-week Stockout Risk"),
            html.P(
                "This page summarizes current stock, weeks of cover, lead time, and "
                "modelled 4-week stockout risk for each essential medicine.",
                className="text-muted",
            ),
            html.Hr(),
            layout_overview_cards(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select product"),
                            dcc.Dropdown(
                                id="ew-product-dropdown",
                                options=product_options,
                                value=PRODUCTS[0],
                                clearable=False,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Current stock on hand"),
                                dbc.CardBody(html.H3(id="ew-kpi-stock")),
                            ]
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Weeks of stock (approx.)"),
                                dbc.CardBody(html.H3(id="ew-kpi-weeks")),
                            ]
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Lead time"),
                                dbc.CardBody(html.H3(id="ew-kpi-leadtime")),
                            ]
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Reorder quantity (lead-time aware)"),
                                dbc.CardBody(html.H3(id="ew-kpi-reorder")),
                            ]
                        ),
                        md=3,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Alert(
                            id="ew-risk-alert",
                            color="secondary",
                            className="w-100",
                        ),
                        md=8,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Projected stockout date"),
                                dbc.CardBody(html.H4(id="ew-kpi-stockout-date")),
                            ]
                        ),
                        md=4,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="ew-risk-bar"), md=12),
                ]
            ),
        ]
    )


def layout_seasonality_page():
    return html.Div(
        [
            html.H3("3. Seasonality Insights – Disease & Demand Patterns"),
            html.P(
                "Average weekly demand by week-of-year helps align procurement with "
                "seasonal patterns (e.g., malaria, diarrhoeal disease, NCDs).",
                className="text-muted",
            ),
            html.Hr(),
            layout_overview_cards(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select product"),
                            dcc.Dropdown(
                                id="seasonality-product-dropdown",
                                options=product_options,
                                value="Artemether-Lumefantrine (AL)"
                                if "Artemether-Lumefantrine (AL)" in PRODUCTS
                                else PRODUCTS[0],
                                clearable=False,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="seasonality-figure"), md=8),
                    dbc.Col(
                        [
                            html.H5("How to interpret this chart"),
                            html.Ul(
                                [
                                    html.Li(
                                        "Peaks indicate periods of higher demand – "
                                        "for example, malaria season or rainy season."
                                    ),
                                    html.Li(
                                        "Procurement teams can align framework contracts "
                                        "and buffer stocks with these patterns."
                                    ),
                                    html.Li(
                                        "The same logic can be extended to facility-level "
                                        "or regional seasonality when real LMIS data is used."
                                    ),
                                ]
                            ),
                        ],
                        md=4,
                    ),
                ]
            ),
        ]
    )


# --------------------------------------------------
# ROUTING CALLBACK – CHOOSE PAGE BY URL
# --------------------------------------------------
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    if pathname == "/forecasting":
        return layout_forecasting_page()
    elif pathname == "/seasonality":
        return layout_seasonality_page()
    elif pathname == "/early-warning" or pathname == "/":
        # default page
        return layout_early_warning_page()
    else:
        return layout_early_warning_page()


# --------------------------------------------------
# FORECASTING CALLBACKS
# --------------------------------------------------
@app.callback(
    Output("forecast-demand-figure", "figure"),
    Output("forecast-stock-figure", "figure"),
    Input("forecast-product-dropdown", "value"),
)
def update_forecast_page(product_name):
    logger.info("Updating forecasting view for product: %s", product_name)
    demand_fig = make_forecast_figure(product_name)
    stock_fig = make_stock_timeseries_figure(product_name)
    return demand_fig, stock_fig


# --------------------------------------------------
# EARLY WARNING CALLBACKS
# --------------------------------------------------
@app.callback(
    Output("ew-kpi-stock", "children"),
    Output("ew-kpi-weeks", "children"),
    Output("ew-kpi-leadtime", "children"),
    Output("ew-kpi-reorder", "children"),
    Output("ew-kpi-stockout-date", "children"),
    Output("ew-risk-alert", "children"),
    Output("ew-risk-alert", "color"),
    Output("ew-risk-bar", "figure"),
    Input("ew-product-dropdown", "value"),
)
def update_early_warning(product_name):
    logger.info("Updating early-warning panel for product: %s", product_name)
    row = current_status[current_status["product_name"] == product_name].iloc[0]

    stock = row["closing_stock"]
    weeks = row.get("projected_weeks_to_stockout", np.nan)
    lead = row.get("lead_time_weeks", np.nan)
    reorder = row.get("reorder_qty_leadtime", np.nan)
    stockout_date = row.get("projected_stockout_date", pd.NaT)
    risk = row.get("risk_4w", np.nan)
    band = row.get("risk_band", "Unknown")
    alert_message = row.get("alert_message", "")

    # KPI display strings
    stock_text = f"{stock:,.0f} units"
    weeks_text = "N/A"
    if pd.notna(weeks) and np.isfinite(weeks):
        weeks_text = f"{weeks:.1f} weeks"

    lead_text = f"{lead:.0f} weeks" if pd.notna(lead) else "N/A"
    reorder_text = f"{reorder:,.0f} units" if pd.notna(reorder) else "0"

    if pd.isna(stockout_date):
        stockout_text = "N/A"
    else:
        stockout_text = stockout_date.strftime("%d %b %Y")

    # Alert colour based on risk band
    if band == "High":
        alert_color = "danger"
    elif band == "Medium":
        alert_color = "warning"
    elif band == "Low":
        alert_color = "success"
    else:
        alert_color = "secondary"

    risk_bar_fig = make_risk_bar_figure(selected_product=product_name)

    # Add concise header to alert
    prefix = ""
    if pd.notna(risk):
        prefix = f"[{band} risk – {risk:.1%} in next 4 weeks] "

    alert_full = prefix + str(alert_message)

    return (
        stock_text,
        weeks_text,
        lead_text,
        reorder_text,
        stockout_text,
        alert_full,
        alert_color,
        risk_bar_fig,
    )


# --------------------------------------------------
# SEASONALITY CALLBACK
# --------------------------------------------------
@app.callback(
    Output("seasonality-figure", "figure"),
    Input("seasonality-product-dropdown", "value"),
)
def update_seasonality_page(product_name):
    logger.info("Updating seasonality view for product: %s", product_name)
    return make_seasonality_figure(product_name)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    # For local debugging / development
    logger.info("Running Dash app in debug mode on port 8050.")
    app.run(host="0.0.0.0", port=8050, debug=True)
