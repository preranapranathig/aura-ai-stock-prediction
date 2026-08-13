import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from live_prediction import generate_live_prediction
from stock_config import STOCKS


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AURA AI | Intelligent Stock Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(35, 80, 120, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(80, 45, 120, 0.14),
                transparent 28%
            ),
            #070b12;
    }

    .main {
        padding-top: 1rem;
    }

    header[data-testid="stHeader"] {
        background: rgba(7, 11, 18, 0.85);
    }

    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #090e17 0%,
                #0b111c 100%
            );
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] h1 {
        letter-spacing: 2px;
    }

    /* =========================
       MAIN BRAND
       ========================= */

    .brand {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 5px;
        color: #f5f7fa;
        margin-bottom: 0;
    }

    .brand-subtitle {
        color: #7f8da3;
        font-size: 0.82rem;
        letter-spacing: 2px;
        margin-top: -5px;
    }

    .status-online {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 30px;
        background: rgba(35, 197, 94, 0.10);
        border: 1px solid rgba(35, 197, 94, 0.30);
        color: #4ade80;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1px;
    }

    /* =========================
       HERO
       ========================= */

    .hero {
        padding: 25px 28px;
        border-radius: 20px;
        background:
            linear-gradient(
                135deg,
                rgba(18, 28, 43, 0.96),
                rgba(12, 17, 28, 0.96)
            );
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow:
            0 20px 60px rgba(0,0,0,0.25);
        margin-top: 18px;
        margin-bottom: 20px;
    }

    .hero-label {
        color: #7f8da3;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .hero-title {
        font-size: 2.0rem;
        font-weight: 750;
        margin-top: 5px;
        color: #f7f9fc;
    }

    .hero-ticker {
        color: #7f8da3;
        font-size: 0.9rem;
        margin-top: 3px;
    }

    /* =========================
       METRIC CARDS
       ========================= */

    .metric-card {
        background:
            linear-gradient(
                145deg,
                rgba(20, 29, 43, 0.96),
                rgba(12, 18, 28, 0.96)
            );
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 17px;
        padding: 19px;
        min-height: 130px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.18);
    }

    .metric-label {
        color: #7f8da3;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1.6px;
    }

    .metric-value {
        color: #f5f7fa;
        font-size: 1.55rem;
        font-weight: 750;
        margin-top: 10px;
    }

    .metric-small {
        color: #7f8da3;
        font-size: 0.74rem;
        margin-top: 5px;
    }

    /* =========================
       SIGNAL
       ========================= */

    .signal-card {
        padding: 22px;
        border-radius: 18px;
        background:
            linear-gradient(
                135deg,
                rgba(20,29,43,0.96),
                rgba(10,16,25,0.96)
            );
        border: 1px solid rgba(255,255,255,0.08);
    }

    .signal-positive {
        color: #4ade80;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .signal-negative {
        color: #fb7185;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .signal-neutral {
        color: #fbbf24;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: 1px;
    }

    /* =========================
       SECTION HEADERS
       ========================= */

    .section-title {
        color: #e7ebf2;
        font-size: 1.0rem;
        font-weight: 750;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .section-line {
        height: 1px;
        background: linear-gradient(
            90deg,
            rgba(255,255,255,0.15),
            transparent
        );
        margin-bottom: 18px;
    }

    /* =========================
       INFO PANEL
       ========================= */

    .info-panel {
        padding: 18px;
        border-radius: 16px;
        background: rgba(15,22,34,0.75);
        border: 1px solid rgba(255,255,255,0.06);
    }

    .info-key {
        color: #718096;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .info-value {
        color: #e9edf4;
        font-size: 0.92rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align: center;
        padding: 28px 0 10px;
        color: #536176;
        font-size: 0.72rem;
        letter-spacing: 1px;
    }

    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.10);
        min-height: 45px;
    }

    /* =========================
       DATAFRAME
       ========================= */

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_stock_file(stock_name):
    """
    Locate historical stock CSV.
    """

    filename = stock_name.lower().replace(" ", "_")

    candidates = [
        BASE_DIR / "data" / "stocks" / f"{filename}.csv",
        BASE_DIR / "data" / f"{filename}_stock_data.csv",
        BASE_DIR / "data" / f"{filename}.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def format_currency(value, currency):
    """
    Format price according to stock currency.
    """

    if currency == "₹":
        return f"₹{value:,.2f}"

    return f"${value:,.2f}"


def get_market_flag(market):
    if market == "India":
        return "🇮🇳"

    return "🇺🇸"


def calculate_risk(uncertainty, price):

    if price <= 0:
        return "UNKNOWN"

    uncertainty_pct = (uncertainty / price) * 100

    if uncertainty_pct < 1:
        return "LOW"

    elif uncertainty_pct < 2.5:
        return "MODERATE"

    elif uncertainty_pct < 5:
        return "HIGH"

    return "VERY HIGH"


def risk_message(risk):

    if risk == "LOW":
        return "Model shows relatively stable prediction behavior."

    if risk == "MODERATE":
        return "Prediction has moderate uncertainty. Monitor market conditions."

    if risk == "HIGH":
        return "Prediction uncertainty is elevated."

    if risk == "VERY HIGH":
        return "Prediction uncertainty is very high."

    return "Risk information unavailable."


def find_metrics_file(stock_name):

    filename = stock_name.lower().replace(" ", "_")

    candidates = [
        BASE_DIR / "results" / f"{filename}_metrics.csv",
        BASE_DIR / "results" / f"{filename}_model_metrics.csv",
        BASE_DIR / "results" / f"{filename}_metrics.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_stock_data(stock_name):

    file_path = get_stock_file(stock_name)

    if file_path is None:
        return None

    try:

        df = pd.read_csv(file_path)

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            df = df.sort_values("Date")

        return df

    except Exception:

        return None


def create_historical_chart(df):

    try:

        import plotly.graph_objects as go

        if df is None or "Close" not in df.columns:
            return None

        plot_df = df.tail(180).copy()

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"],
                y=plot_df["Close"],
                mode="lines",
                name="Market Price",
                line=dict(
                    width=2
                ),
                hovertemplate=
                "%{x|%d %b %Y}<br>"
                "Price: %{y:.2f}<extra></extra>"
            )
        )

        fig.update_layout(

            height=440,

            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            font=dict(
                color="#9aa8bb"
            ),

            xaxis=dict(
                showgrid=False,
                zeroline=False
            ),

            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                zeroline=False
            ),

            legend=dict(
                bgcolor="rgba(0,0,0,0)"
            )
        )

        return fig

    except Exception:

        return None


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns(
    [7, 2]
)

with header_left:

    st.markdown(
        """
        <div class="brand">AURA AI</div>
        <div class="brand-subtitle">
            ADAPTIVE UNCERTAINTY • RISK • ANALYTICS
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:

    st.markdown(
        """
        <div style="text-align:right;margin-top:12px;">
            <span class="status-online">
                ● AI ENGINE ONLINE
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR — STOCK SELECTION
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size:1.35rem;
        font-weight:800;
        letter-spacing:2px;
        margin-bottom:4px;
    ">
        MARKET CONTROL
    </div>

    <div style="
        color:#68778c;
        font-size:0.72rem;
        letter-spacing:1px;
        margin-bottom:20px;
    ">
        SELECT AI MODEL
    </div>
    """,
    unsafe_allow_html=True
)


# Group stocks by market

indian_stocks = [
    name
    for name, info in STOCKS.items()
    if info.get("market") == "India"
]

us_stocks = [
    name
    for name, info in STOCKS.items()
    if info.get("market") == "USA"
]


market = st.sidebar.radio(
    "Market",
    ["🇮🇳 India", "🇺🇸 United States"],
    horizontal=False
)


if market.startswith("🇮🇳"):

    available_stocks = indian_stocks

else:

    available_stocks = us_stocks


selected_stock = st.sidebar.selectbox(
    "Stock",
    available_stocks,
    index=0
)


config = STOCKS[selected_stock]

currency = config.get(
    "currency",
    "₹"
)

ticker = config.get(
    "ticker",
    "N/A"
)

company_name = config.get(
    "name",
    selected_stock
)


st.sidebar.divider()


st.sidebar.markdown(
    f"""
    <div class="info-panel">

        <div class="info-key">Selected Company</div>
        <div class="info-value">{company_name}</div>

        <br>

        <div class="info-key">Ticker</div>
        <div class="info-value">{ticker}</div>

        <br>

        <div class="info-key">Market</div>
        <div class="info-value">
            {get_market_flag(config.get("market"))}
            {config.get("market")}
        </div>

        <br>

        <div class="info-key">AI Architecture</div>
        <div class="info-value">
            LSTM + Monte Carlo Dropout
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown(
    """
    <div style="
        margin-top:20px;
        color:#536176;
        font-size:0.68rem;
        line-height:1.6;
    ">
        Predictions are generated from the
        selected stock's dedicated trained model.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">

        <div class="hero-label">
            AI MARKET INTELLIGENCE
        </div>

        <div class="hero-title">
            {get_market_flag(config.get("market"))}
            {company_name}
        </div>

        <div class="hero-ticker">
            {ticker}
            &nbsp; • &nbsp;
            {config.get("market")}
            &nbsp; • &nbsp;
            Dedicated LSTM Model
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

stock_data = load_stock_data(selected_stock)


# ============================================================
# DATA STATUS
# ============================================================

if stock_data is not None:

    last_date = "Unavailable"

    if "Date" in stock_data.columns:

        valid_dates = stock_data["Date"].dropna()

        if len(valid_dates) > 0:

            last_date = valid_dates.iloc[-1].strftime(
                "%d %b %Y"
            )

    data_rows = len(stock_data)

else:

    last_date = "Unavailable"
    data_rows = 0


# ============================================================
# RUN LIVE PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">Live AI Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-line"></div>',
    unsafe_allow_html=True
)


run_prediction = st.button(
    f"⚡ RUN {selected_stock.upper()} AI ANALYSIS",
    use_container_width=True
)


if run_prediction:

    with st.spinner(
        f"Running {selected_stock} LSTM + uncertainty engine..."
    ):

        try:

            live_result = generate_live_prediction(
                selected_stock
            )

            st.session_state["live_result"] = live_result

            st.session_state["live_stock"] = selected_stock

            st.session_state["prediction_time"] = time.strftime(
                "%d %b %Y • %H:%M:%S"
            )

        except Exception as e:

            st.session_state["live_error"] = str(e)


# ============================================================
# RETRIEVE PREDICTION
# ============================================================

live_result = st.session_state.get(
    "live_result"
)

live_stock = st.session_state.get(
    "live_stock"
)

live_error = st.session_state.pop(
    "live_error",
    None
)


if live_error:

    st.error(
        f"AI prediction failed: {live_error}"
    )


# Reset if user changes stock

if live_stock != selected_stock:

    live_result = None


# ============================================================
# LIVE RESULTS
# ============================================================

if live_result is not None:

    latest_price = float(
        live_result.get(
            "latest_price",
            0
        )
    )

    predicted_price = float(
        live_result.get(
            "predicted_price",
            0
        )
    )

    percentage_change = float(
        live_result.get(
            "percentage_change",
            0
        )
    )

    uncertainty = float(
        live_result.get(
            "uncertainty",
            0
        )
    )

    lower_bound = float(
        live_result.get(
            "lower_bound",
            predicted_price - uncertainty
        )
    )

    upper_bound = float(
        live_result.get(
            "upper_bound",
            predicted_price + uncertainty
        )
    )

    signal = live_result.get(
        "signal",
        "Neutral"
    )

    risk = calculate_risk(
        uncertainty,
        predicted_price
    )


    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    cols = st.columns(4)

    with cols[0]:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Live Market Price
                </div>

                <div class="metric-value">
                    {format_currency(latest_price, currency)}
                </div>

                <div class="metric-small">
                    Latest market observation
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with cols[1]:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    AI Forecast
                </div>

                <div class="metric-value">
                    {format_currency(predicted_price, currency)}
                </div>

                <div class="metric-small">
                    Next predicted price
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with cols[2]:

        change_symbol = (
            "▲"
            if percentage_change >= 0
            else "▼"
        )

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Expected Movement
                </div>

                <div class="metric-value">
                    {change_symbol}
                    {percentage_change:+.2f}%
                </div>

                <div class="metric-small">
                    AI forecast vs current price
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with cols[3]:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    AI Uncertainty
                </div>

                <div class="metric-value">
                    ±{format_currency(uncertainty, currency)}
                </div>

                <div class="metric-small">
                    Monte Carlo prediction spread
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # SIGNAL + RISK
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">AI Decision Layer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True
    )


    signal_col, risk_col, range_col = st.columns(
        [1.1, 1, 1.5]
    )


    with signal_col:

        if signal.lower() == "positive":

            signal_class = "signal-positive"
            signal_text = "▲ POSITIVE"

        elif signal.lower() == "negative":

            signal_class = "signal-negative"
            signal_text = "▼ NEGATIVE"

        else:

            signal_class = "signal-neutral"
            signal_text = "● NEUTRAL"


        st.markdown(
            f"""
            <div class="signal-card">

                <div class="metric-label">
                    AI SIGNAL
                </div>

                <div class="{signal_class}">
                    {signal_text}
                </div>

                <div class="metric-small">
                    Based on model forecast
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with risk_col:

        risk_symbol = {
            "LOW": "◉",
            "MODERATE": "◐",
            "HIGH": "◑",
            "VERY HIGH": "◉"
        }.get(
            risk,
            "●"
        )

        st.markdown(
            f"""
            <div class="signal-card">

                <div class="metric-label">
                    MODEL RISK
                </div>

                <div style="
                    color:#f5f7fa;
                    font-size:1.45rem;
                    font-weight:800;
                    margin-top:8px;
                ">
                    {risk_symbol} {risk}
                </div>

                <div class="metric-small">
                    {risk_message(risk)}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with range_col:

        st.markdown(
            f"""
            <div class="signal-card">

                <div class="metric-label">
                    95% PREDICTION RANGE
                </div>

                <div style="
                    color:#f5f7fa;
                    font-size:1.25rem;
                    font-weight:800;
                    margin-top:8px;
                ">
                    {format_currency(lower_bound, currency)}
                    &nbsp; → &nbsp;
                    {format_currency(upper_bound, currency)}
                </div>

                <div class="metric-small">
                    Uncertainty interval produced by AI engine
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # FORECAST VISUAL
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Market Trajectory</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True
    )


    try:

        import plotly.graph_objects as go

        if (
            stock_data is not None
            and "Close" in stock_data.columns
            and "Date" in stock_data.columns
        ):

            chart_df = stock_data.tail(180).copy()

            fig = go.Figure()


            # Historical price

            fig.add_trace(
                go.Scatter(
                    x=chart_df["Date"],
                    y=chart_df["Close"],
                    mode="lines",
                    name="Historical Price",
                    line=dict(
                        width=2.5
                    )
                )
            )


            # Forecast point

            if len(chart_df) > 0:

                last_date = chart_df["Date"].iloc[-1]

                future_date = (
                    last_date
                    + pd.Timedelta(days=1)
                )

                fig.add_trace(
                    go.Scatter(
                        x=[
                            last_date,
                            future_date
                        ],
                        y=[
                            latest_price,
                            predicted_price
                        ],
                        mode="lines+markers",
                        name="AI Forecast",
                        line=dict(
                            width=3,
                            dash="dot"
                        ),
                        marker=dict(
                            size=9
                        )
                    )
                )


                # Upper uncertainty

                fig.add_trace(
                    go.Scatter(
                        x=[
                            future_date
                        ],
                        y=[
                            upper_bound
                        ],
                        mode="markers",
                        name="Upper Bound",
                        marker=dict(
                            size=7
                        )
                    )
                )


                # Lower uncertainty

                fig.add_trace(
                    go.Scatter(
                        x=[
                            future_date
                        ],
                        y=[
                            lower_bound
                        ],
                        mode="markers",
                        name="Lower Bound",
                        marker=dict(
                            size=7
                        )
                    )
                )


            fig.update_layout(

                height=470,

                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                font=dict(
                    color="#9aa8bb"
                ),

                hovermode="x unified",

                xaxis=dict(
                    showgrid=False,
                    zeroline=False
                ),

                yaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.05)",
                    zeroline=False,
                    tickprefix=currency
                ),

                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "Historical chart data is unavailable."
            )

    except ImportError:

        st.warning(
            "Plotly is required for the advanced chart."
        )


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-line"></div>',
        unsafe_allow_html=True
    )


    info1, info2, info3, info4 = st.columns(4)


    with info1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    MODEL
                </div>

                <div class="metric-value"
                     style="font-size:1.05rem;">
                    LSTM
                </div>

                <div class="metric-small">
                    Long Short-Term Memory
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with info2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    UNCERTAINTY
                </div>

                <div class="metric-value"
                     style="font-size:1.05rem;">
                    MC DROPOUT
                </div>

                <div class="metric-small">
                    Monte Carlo simulations
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with info3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    DATA POINTS
                </div>

                <div class="metric-value"
                     style="font-size:1.05rem;">
                    {data_rows:,}
                </div>

                <div class="metric-small">
                    Historical observations
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with info4:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    DATA THROUGH
                </div>

                <div class="metric-value"
                     style="font-size:1.05rem;">
                    {last_date}
                </div>

                <div class="metric-small">
                    Latest stored observation
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


else:

    # ========================================================
    # BEFORE FIRST PREDICTION
    # ========================================================

    st.markdown(
        """
        <div class="signal-card"
             style="text-align:center;padding:45px;">

            <div style="
                font-size:2.5rem;
                margin-bottom:10px;
            ">
                ◈
            </div>

            <div style="
                font-size:1.25rem;
                font-weight:750;
                color:#f5f7fa;
            ">
                AI Engine Ready
            </div>

            <div style="
                color:#718096;
                margin-top:8px;
                font-size:0.85rem;
            ">
                Select a stock and run the AI analysis
                to generate a live prediction.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HISTORICAL DATA
# ============================================================

st.markdown(
    '<div class="section-title">Historical Market Data</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-line"></div>',
    unsafe_allow_html=True
)


if stock_data is not None:

    # Price chart

    try:

        chart = create_historical_chart(
            stock_data
        )

        if chart is not None:

            st.plotly_chart(
                chart,
                use_container_width=True
            )

    except Exception:

        pass


    with st.expander(
        f"View {selected_stock} raw market data"
    ):

        st.dataframe(
            stock_data.tail(30),
            use_container_width=True,
            hide_index=True
        )

else:

    st.warning(
        f"No historical dataset found for {selected_stock}."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        AURA AI • INTELLIGENT STOCK ANALYTICS

        <br>

        LSTM Forecasting • Monte Carlo Dropout •
        Uncertainty-Aware Prediction

        <br><br>

        Research & Educational Use •
        Not Financial Advice

    </div>
    """,
    unsafe_allow_html=True
)