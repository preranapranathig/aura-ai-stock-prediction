import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import textwrap
import hashlib
import hmac
import secrets
import re
import html
import json
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

st.markdown("""
<style>

/* =========================================================
   AURA AI — PROFESSIONAL DARK FINTECH TERMINAL
   ========================================================= */

.stApp {
    background:
        radial-gradient(circle at 85% 0%, rgba(0, 200, 255, 0.07), transparent 25%),
        radial-gradient(circle at 10% 15%, rgba(60, 100, 180, 0.06), transparent 28%),
        #070a10 !important;
    color: #e8eef7 !important;
}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 3rem !important;
    max-width: 1480px !important;
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #0a0e15 0%, #070a10 100%) !important;
    border-right: 1px solid rgba(148, 163, 184, 0.10);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.7rem;
}

section[data-testid="stSidebar"] * {
    color: #d7e0eb;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* ---------- SIDEBAR INFO ---------- */

.info-panel {
    padding: 18px 16px;
    margin-top: 6px;
    background: linear-gradient(145deg, #0e141f, #090d15);
    border: 1px solid rgba(148, 163, 184, 0.11);
    border-radius: 12px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.22);
}

.info-key {
    color: #64748b !important;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    margin-bottom: 5px;
}

.info-value {
    color: #e8eef7 !important;
    font-size: 0.82rem;
    font-weight: 600;
    line-height: 1.45;
}

/* ---------- HEADER ---------- */

.brand {
    font-size: 1.75rem;
    font-weight: 850;
    letter-spacing: 0.10em;
    color: #f8fafc;
    line-height: 1.1;
}

.brand-subtitle {
    margin-top: 8px;
    color: #718096;
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.11em;
}

.status-online {
    display: inline-flex;
    align-items: center;
    padding: 7px 12px;
    border-radius: 999px;
    color: #67e8f9 !important;
    background: rgba(34, 211, 238, 0.07);
    border: 1px solid rgba(34, 211, 238, 0.18);
    font-size: 0.68rem;
    font-weight: 750;
    letter-spacing: 0.06em;
}

/* ---------- HERO ---------- */

.hero {
    position: relative;
    margin: 10px 0 24px 0;
    padding: 26px 28px;
    background:
        linear-gradient(135deg, rgba(14, 22, 35, 0.98), rgba(8, 12, 19, 0.98));
    border: 1px solid rgba(148, 163, 184, 0.11);
    border-radius: 16px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.24);
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    width: 3px;
    height: 100%;
    background: linear-gradient(180deg, #22d3ee, #3b82f6);
}

.hero-label {
    color: #5eead4;
    font-size: 0.67rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.hero-title {
    margin-top: 8px;
    color: #f8fafc;
    font-size: 1.65rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}

.hero-ticker {
    margin-top: 8px;
    color: #718096;
    font-size: 0.78rem;
    font-weight: 550;
}

/* ---------- SECTION TITLES ---------- */

.section-title {
    margin-top: 30px;
    margin-bottom: 10px;
    color: #f8fafc;
    font-size: 0.88rem;
    font-weight: 800;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}

.section-line {
    height: 1px;
    margin-bottom: 18px;
    background: linear-gradient(
        90deg,
        rgba(34, 211, 238, 0.45),
        rgba(148, 163, 184, 0.10),
        transparent
    );
}

/* ---------- BUTTON ---------- */

.stButton > button {
    width: 100%;
    min-height: 46px;
    border-radius: 10px;
    border: 1px solid rgba(34, 211, 238, 0.25);
    background: linear-gradient(135deg, #111a29, #0c1421);
    color: #dffbff !important;
    font-weight: 750;
    letter-spacing: 0.04em;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: rgba(34, 211, 238, 0.60);
    background: linear-gradient(135deg, #142337, #0d1929);
    box-shadow: 0 0 24px rgba(34, 211, 238, 0.10);
    transform: translateY(-1px);
}

/* ---------- METRIC CARDS ---------- */

.metric-card {
    min-height: 112px;
    padding: 19px 20px;
    background: linear-gradient(145deg, #0e1521, #090e16);
    border: 1px solid rgba(148, 163, 184, 0.11);
    border-radius: 14px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.20);
    transition: all 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(34, 211, 238, 0.25);
    box-shadow: 0 16px 36px rgba(0,0,0,0.30);
}

.metric-label {
    color: #718096 !important;
    font-size: 0.65rem;
    font-weight: 750;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.metric-value {
    margin-top: 9px;
    color: #f8fafc !important;
    font-size: 1.50rem;
    font-weight: 800;
    letter-spacing: -0.025em;
}

.metric-small {
    margin-top: 6px;
    color: #607089 !important;
    font-size: 0.70rem;
}

/* ---------- DECISION CARDS ---------- */

.signal-card {
    min-height: 105px;
    padding: 21px;
    background: linear-gradient(145deg, #0e1521, #090e16);
    border: 1px solid rgba(148, 163, 184, 0.11);
    border-radius: 15px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.22);
}

.signal-positive {
    margin-top: 9px;
    color: #4ade80 !important;
    font-size: 1.20rem;
    font-weight: 800;
}

.signal-negative {
    margin-top: 9px;
    color: #fb7185 !important;
    font-size: 1.20rem;
    font-weight: 800;
}

.signal-neutral {
    margin-top: 9px;
    color: #67e8f9 !important;
    font-size: 1.20rem;
    font-weight: 800;
}

/* ---------- SELECTBOX ---------- */

div[data-baseweb="select"] > div {
    background: #0d131e !important;
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    border-radius: 9px !important;
    color: #f8fafc !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: rgba(34, 211, 238, 0.35) !important;
}

/* ---------- RADIO ---------- */

div[data-testid="stRadio"] label {
    color: #aeb9c8 !important;
}

div[data-testid="stRadio"] label:hover {
    color: #ffffff !important;
}

/* ---------- EXPANDER ---------- */

.streamlit-expanderHeader {
    background: #0d131e !important;
    border: 1px solid rgba(148, 163, 184, 0.10) !important;
    border-radius: 10px !important;
    color: #dce5ef !important;
}

/* ---------- DATAFRAME ---------- */

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.10);
}

/* ---------- ALERTS ---------- */

div[data-testid="stAlert"] {
    background: #0d131e !important;
    border-radius: 10px !important;
    border: 1px solid rgba(148, 163, 184, 0.10);
}

/* ---------- PLOTLY ---------- */

.js-plotly-plot {
    border-radius: 14px;
}

/* ---------- FOOTER ---------- */

.footer {
    margin-top: 45px;
    padding: 28px;
    text-align: center;
    color: #56657a;
    font-size: 0.70rem;
    line-height: 1.6;
    border-top: 1px solid rgba(148, 163, 184, 0.08);
}

.footer strong {
    color: #aeb9c8;
}

/* ---------- SCROLLBAR ---------- */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #070a10;
}

::-webkit-scrollbar-thumb {
    background: #263244;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #3b4d63;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}



/* ---------- AUTHENTICATION ---------- */
.auth-page {
    min-height: 82vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 35px 15px;
}
.auth-card {
    width: min(440px, 100%);
    padding: 34px;
    background: linear-gradient(145deg, #0e1521, #090e16);
    border: 1px solid rgba(148,163,184,0.13);
    border-radius: 20px;
    box-shadow: 0 24px 70px rgba(0,0,0,0.42);
}
.auth-logo {
    width: 66px;
    height: 66px;
    margin: 0 auto 18px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 17px;
    background: linear-gradient(135deg,#122238,#0c1522);
    border: 1px solid rgba(34,211,238,0.28);
    color: #67e8f9;
    font-size: 1.25rem;
    font-weight: 900;
    letter-spacing: .08em;
}
.auth-title {
    text-align: center;
    color: #f8fafc;
    font-size: 1.65rem;
    font-weight: 850;
    letter-spacing: .10em;
}
.auth-subtitle {
    margin-top: 7px;
    margin-bottom: 26px;
    text-align: center;
    color: #718096;
    font-size: .68rem;
    font-weight: 650;
    letter-spacing: .11em;
    text-transform: uppercase;
}
.auth-footer {
    margin-top: 18px;
    text-align: center;
    color: #56657a;
    font-size: .68rem;
    line-height: 1.5;
}

/* ---------- COMPANY LOGO ---------- */

.company-logo {
    width: 54px;
    height: 54px;
    min-width: 54px;
    border-radius: 13px;
    object-fit: contain;
    background: #f8fafc;
    border: 1px solid rgba(255,255,255,0.12);
    padding: 7px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.28);
}

.company-logo-small {
    width: 42px;
    height: 42px;
    min-width: 42px;
    border-radius: 11px;
    object-fit: contain;
    background: #f8fafc;
    border: 1px solid rgba(255,255,255,0.12);
    padding: 6px;
}

.company-logo-fallback {
    width: 54px;
    height: 54px;
    min-width: 54px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #122238, #0c1522);
    border: 1px solid rgba(34,211,238,0.24);
    color: #67e8f9;
    font-size: 0.82rem;
    font-weight: 850;
}

.company-identity {
    display: flex;
    align-items: center;
    gap: 14px;
}

.company-identity-text {
    min-width: 0;
}

.company-identity-name {
    color: #f8fafc;
    font-size: 0.96rem;
    font-weight: 780;
    line-height: 1.25;
}

.company-identity-meta {
    margin-top: 4px;
    color: #718096;
    font-size: 0.68rem;
    font-weight: 650;
}

.hero-company {
    display: flex;
    align-items: center;
    gap: 16px;
}

.hero-company .company-logo,
.hero-company .company-logo-fallback {
    width: 62px;
    height: 62px;
    min-width: 62px;
}

/* ---------- MOBILE ---------- */

@media (max-width: 900px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .brand {
        font-size: 1.45rem;
    }

    .hero {
        padding: 21px;
    }

    .hero-title {
        font-size: 1.35rem;
    }
}
</style>
""", unsafe_allow_html=True)




# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_stock_file(stock_name):
    '''
    Locate historical stock CSV.
    '''

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
    '''
    Format price according to stock currency.
    '''

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


def get_last_data_date(df):
    """Return the most recent Date in a stock dataframe, or None."""
    if df is None or "Date" not in df.columns:
        return None
    valid_dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    if not len(valid_dates):
        return None
    return valid_dates.max()


def is_data_stale(last_date, max_stale_trading_days=1):
    """True if last_date is more than max_stale_trading_days business days
    behind today. Weekends are not counted as staleness, so a Friday close
    on a Saturday/Sunday is NOT flagged as stale."""
    if last_date is None or pd.isna(last_date):
        return True
    today = pd.Timestamp.now().normalize()
    if last_date.normalize() >= today:
        return False
    missing_bdays = pd.bdate_range(start=last_date + pd.Timedelta(days=1), end=today)
    return len(missing_bdays) > max_stale_trading_days


def get_stock_save_path(stock_name):
    """Where a refreshed CSV should be written. Reuses an existing file's
    location if one exists, otherwise defaults to data/stocks/<name>.csv."""
    existing = get_stock_file(stock_name)
    if existing is not None:
        return existing
    filename = stock_name.lower().replace(" ", "_")
    return BASE_DIR / "data" / "stocks" / f"{filename}.csv"


def refresh_stock_data(stock_name, ticker, existing_df):
    """Fetch any missing recent rows via yfinance and merge them into
    existing_df. Returns (updated_df, error_message). error_message is
    None on success (including 'nothing new to fetch')."""
    try:
        import yfinance as yf
    except ImportError:
        return existing_df, "yfinance is not installed. Run: pip install yfinance"

    try:
        last_date = get_last_data_date(existing_df)
        start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if last_date is not None else None

        fresh = yf.download(ticker, start=start, progress=False, auto_adjust=False)
        if fresh is None or fresh.empty:
            return existing_df, None  # already up to date, not an error

        fresh = fresh.reset_index()
        if isinstance(fresh.columns, pd.MultiIndex):
            fresh.columns = [c[0] if isinstance(c, tuple) else c for c in fresh.columns]
        keep_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in fresh.columns]
        fresh = fresh[keep_cols]
        fresh["Date"] = pd.to_datetime(fresh["Date"])

        if existing_df is not None and len(existing_df):
            merged = existing_df.copy()
            merged["Date"] = pd.to_datetime(merged["Date"])
            merged = pd.concat([merged, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset="Date", keep="last").sort_values("Date")
        else:
            merged = fresh.sort_values("Date")

        return merged.reset_index(drop=True), None
    except Exception as exc:
        return existing_df, str(exc)


def save_stock_data(stock_name, df):
    """Persist a refreshed dataframe back to its CSV file."""
    path = get_stock_save_path(stock_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
    """Heuristic confidence score derived from uncertainty percentage.
    This is NOT a calibrated statistic — it's a simple, transparent
    mapping so the dashboard has something meaningful to show. Replace
    with a real backtested R² / coverage score once available."""
    conf = 100 - (uncertainty_pct * 6)
    return max(50.0, min(99.0, conf))


def get_confidence_pct(uncertainty_pct):
    """Heuristic confidence score derived from uncertainty percentage.
    This is NOT a calibrated statistic — it's a simple, transparent
    mapping so the dashboard has something meaningful to show. Replace
    with a real backtested R² / coverage score once available."""
    conf = 100 - (uncertainty_pct * 6)
    return max(50.0, min(99.0, conf))


def render_confidence_donut(value_pct, label="CONFIDENCE"):
    import plotly.graph_objects as go
    value_pct = max(0.0, min(100.0, value_pct))
    fig = go.Figure(data=[go.Pie(
        values=[value_pct, 100 - value_pct],
        hole=0.76,
        marker=dict(colors=["#22d3ee", "rgba(255,255,255,0.06)"]),
        textinfo="none",
        sort=False,
        direction="clockwise",
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=190,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[
            dict(text=f"{value_pct:.1f}%", x=0.5, y=0.54, font=dict(size=26, color="#f8fafc"), showarrow=False),
            dict(text=label, x=0.5, y=0.38, font=dict(size=10, color="#718096"), showarrow=False),
        ],
    )
    return fig


def render_sparkline(values, color="#22d3ee"):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values, mode="lines", line=dict(width=2.4, color=color),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.10)",
    ))
    fig.update_layout(
        height=70, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def get_timeframe_days(timeframe):
    return {"1W": 7, "1M": 30, "3M": 90, "1Y": 365, "All": None}.get(timeframe, 90)


def find_predictions_file(stock_name):
    """Find backtest file. Prefer .npz (actual/predicted), then .csv."""
    filename = str(stock_name).lower().replace(" ", "_")
    candidates = [
        BASE_DIR / "results" / f"{filename}_predictions.npz",
        BASE_DIR / "results" / f"{filename}_predictions.csv",
        BASE_DIR / "results" / f"{filename}_backtest.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_backtest_df(path):
    """Load Actual/Predicted from npz or csv."""
    if path is None:
        return None

    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        key_map = {k.lower(): k for k in data.files}
        actual_key = key_map.get("actual") or key_map.get("y_true")
        pred_key = key_map.get("predicted") or key_map.get("y_pred")
        if actual_key is None or pred_key is None:
            return None
        actual = np.asarray(data[actual_key]).reshape(-1)
        predicted = np.asarray(data[pred_key]).reshape(-1)
        n = min(len(actual), len(predicted))
        return pd.DataFrame({
            "Actual": actual[:n],
            "Predicted": predicted[:n],
            "Date": list(range(1, n + 1)),
        })

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("actual", "y_true", "true"):
            rename[c] = "Actual"
        elif cl in ("predicted", "y_pred", "pred", "prediction"):
            rename[c] = "Predicted"
        elif cl in ("date", "datetime", "time"):
            rename[c] = "Date"
    df = df.rename(columns=rename)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df




# ============================================================
# ============================================================
# AUTHENTICATION & USER PROFILES
# ============================================================

# Persistent local SQLite user store.
import sqlite3

AUTH_DB = BASE_DIR / "data" / "aura_users.db"
AUTH_DB.parent.mkdir(parents=True, exist_ok=True)


def get_auth_connection():
    conn = sqlite3.connect(AUTH_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    """Create the persistent users table and seed demo accounts."""
    with get_auth_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT DEFAULT '',
                role TEXT DEFAULT 'User',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

        demo_users = [
            (
                "admin",
                hashlib.sha256(
                    "ChangeMe_Admin_2026!".encode("utf-8")
                ).hexdigest(),
                "AURA Administrator",
                "admin@aura-ai.local",
                "",
                "User",
            ),
            (
                "analyst",
                hashlib.sha256(
                    "ChangeMe_Analyst_2026!".encode("utf-8")
                ).hexdigest(),
                "AURA Analyst",
                "analyst@aura-ai.local",
                "",
                "User",
            ),
        ]

        for user in demo_users:
            conn.execute(
                """
                INSERT OR IGNORE INTO users
                (username, password_hash, display_name, email, phone, role)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                user,
            )
        conn.commit()


init_auth_db()


def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_bytes(16)

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        str(password).encode("utf-8"),
        salt,
        200_000,
    )

    return salt.hex() + "$" + derived.hex()


def verify_password_hash(password, stored_hash):
    if "$" in str(stored_hash):
        try:
            salt_hex, digest_hex = str(stored_hash).split("$", 1)
            salt = bytes.fromhex(salt_hex)
        except (ValueError, TypeError):
            return False

        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            str(password).encode("utf-8"),
            salt,
            200_000,
        ).hex()

        return hmac.compare_digest(candidate, digest_hex)

    candidate = hashlib.sha256(
        str(password).encode("utf-8")
    ).hexdigest()

    return hmac.compare_digest(candidate, str(stored_hash))


def get_user(username):
    username_key = str(username).strip().lower()
    if not username_key:
        return None

    with get_auth_connection() as conn:
        row = conn.execute(
            "SELECT username, password_hash, display_name, email, phone, role "
            "FROM users WHERE username = ?",
            (username_key,),
        ).fetchone()

    return dict(row) if row else None


def verify_password(username, password):
    user = get_user(username)

    if not user:
        return False

    return verify_password_hash(
        password,
        user["password_hash"],
    )


def create_user(username, password, display_name, email, phone):
    username = str(username).strip().lower()
    email = str(email).strip().lower()

    if not re.fullmatch(
        r"[a-z0-9._-]{3,30}",
        username,
    ):
        return False, (
            "Username must be 3–30 characters and use "
            "letters, numbers, dot, underscore or hyphen."
        )

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    if not re.fullmatch(
        r"[^@\s]+@[^@\s]+\.[^@\s]+",
        email,
    ):
        return False, "Please enter a valid email address."

    if not str(display_name).strip():
        return False, "Please enter your full name."

    if get_user(username):
        return False, "That username already exists."

    password_hash = hash_password(password)

    try:
        with get_auth_connection() as conn:
            conn.execute(
                """
                INSERT INTO users
                (username, password_hash, display_name, email, phone, role)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    password_hash,
                    str(display_name).strip(),
                    email,
                    str(phone).strip(),
                    "User",
                ),
            )
            conn.commit()

    except sqlite3.IntegrityError as exc:
        if "email" in str(exc).lower():
            return False, "That email address is already registered."
        return False, "That username already exists."

    return True, "Account created successfully."


def update_user_profile(username, display_name, email, phone):
    username = str(username).strip().lower()
    email = str(email).strip().lower()
    display_name = str(display_name).strip()
    phone = str(phone).strip()

    if not display_name:
        return False, "Please enter your full name."

    if not re.fullmatch(
        r"[^@\s]+@[^@\s]+\.[^@\s]+",
        email,
    ):
        return False, "Please enter a valid email address."

    try:
        with get_auth_connection() as conn:
            conn.execute(
                """
                UPDATE users
                SET display_name = ?, email = ?, phone = ?
                WHERE username = ?
                """,
                (display_name, email, phone, username),
            )
            conn.commit()

    except sqlite3.IntegrityError:
        return False, "That email address is already registered."

    return True, "Profile updated successfully."


def reset_user_password(username, email, new_password):
    username_key = str(username).strip().lower()
    email_key = str(email).strip().lower()

    user = get_user(username_key)

    if not user:
        return False, "Unable to verify those account details."

    if not hmac.compare_digest(
        email_key,
        str(user.get("email", "")).lower(),
    ):
        return False, "Unable to verify those account details."

    if len(new_password) < 8:
        return False, "New password must contain at least 8 characters."

    with get_auth_connection() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (hash_password(new_password), username_key),
        )
        conn.commit()

    return True, "Password reset successfully. You can now sign in."


def init_auth_state():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "auth_username" not in st.session_state:
        st.session_state.auth_username = None

    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "login"

    if "profile_editing" not in st.session_state:
        st.session_state.profile_editing = False


def logout():
    st.session_state.authenticated = False
    st.session_state.auth_username = None
    st.session_state.auth_view = "login"
    st.session_state.profile_editing = False


def render_login_page():
    st.markdown(
    '<div class="aura-header-wrap" style="text-align:center;padding-top:40px;margin-bottom:8px;">'
    '<div class="brand" style="justify-content:center;">AURA AI</div>'
    '<div class="brand-subtitle">ADAPTIVE UNCERTAINTY • RISK • ANALYTICS</div>'
    '</div>',
    unsafe_allow_html=True,
)

    _, center, _ = st.columns([1, 2, 1])

    with center:
        st.markdown(
            '<div style="height:12px;"></div>',
            unsafe_allow_html=True,
        )

        if st.session_state.auth_view == "forgot":
            render_forgot_password()
            return

        if st.session_state.auth_view == "signup":
            render_signup()
            return

        with st.form("aura_login_form"):
            st.markdown(
                '<div style="color:#94a3b8;font-size:.72rem;'
                'font-weight:700;letter-spacing:.10em;'
                'text-transform:uppercase;margin-bottom:8px;">'
                'Secure Access</div>',
                unsafe_allow_html=True,
            )

            username = st.text_input(
                "Username",
                placeholder="Enter your username",
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
            )

            submitted = st.form_submit_button(
                "SIGN IN",
                use_container_width=True,
            )

        if submitted:
            username_key = str(username).strip().lower()

            if verify_password(username_key, password):
                st.session_state.authenticated = True
                st.session_state.auth_username = username_key
                st.session_state.auth_view = "login"
                st.rerun()
            else:
                st.error("Invalid username or password.")

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "CREATE PROFILE",
                use_container_width=True,
            ):
                st.session_state.auth_view = "signup"
                st.rerun()

        with c2:
            if st.button(
                "FORGOT PASSWORD?",
                use_container_width=True,
            ):
                st.session_state.auth_view = "forgot"
                st.rerun()

        st.markdown(
            '<div class="auth-footer">'
            'Secure access to AURA AI analytics<br>'
            'Authorized users only'
            '</div>',
            unsafe_allow_html=True,
        )


def render_signup():
    st.markdown(
        '<div class="section-title">Create Your AURA AI Profile</div>',
        unsafe_allow_html=True,
    )

    with st.form("aura_signup_form"):
        full_name = st.text_input(
            "Full Name",
            placeholder="Your name",
        )

        email = st.text_input(
            "Email Address",
            placeholder="you@example.com",
        )

        phone = st.text_input(
            "Phone (optional)",
            placeholder="+91 ...",
        )

        username = st.text_input(
            "Username",
            placeholder="Choose a username",
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Minimum 8 characters",
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Re-enter your password",
        )

        agree = st.checkbox(
            "I agree to use AURA AI for authorized analytics."
        )

        submitted = st.form_submit_button(
            "CREATE ACCOUNT",
            use_container_width=True,
        )

    if submitted:
        if password != confirm_password:
            st.error("Passwords do not match.")
            return

        if not agree:
            st.error("Please accept the usage agreement.")
            return

        success, message = create_user(
            username,
            password,
            full_name,
            email,
            phone,
        )

        if success:
            st.success(message)

            st.session_state.authenticated = True
            st.session_state.auth_username = (
                str(username).strip().lower()
            )
            st.session_state.auth_view = "login"
            st.rerun()
        else:
            st.error(message)

    if st.button(
        "← BACK TO SIGN IN",
        use_container_width=True,
    ):
        st.session_state.auth_view = "login"
        st.rerun()


def render_forgot_password():
    st.markdown(
        '<div class="section-title">Reset Password</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "For this local prototype, password reset is verified using "
        "the username and registered account email. For production, "
        "use a verified email OTP or secure reset link."
    )

    with st.form("aura_reset_form"):
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
        )

        email = st.text_input(
            "Account Email",
            placeholder="Enter your registered email",
        )

        new_password = st.text_input(
            "New Password",
            type="password",
            placeholder="Minimum 8 characters",
        )

        confirm_password = st.text_input(
            "Confirm New Password",
            type="password",
        )

        submitted = st.form_submit_button(
            "RESET PASSWORD",
            use_container_width=True,
        )

    if submitted:
        if new_password != confirm_password:
            st.error("Passwords do not match.")
            return

        success, message = reset_user_password(
            username,
            email,
            new_password,
        )

        if success:
            st.success(message)
            st.session_state.auth_view = "login"
            st.rerun()
        else:
            st.error(message)

    if st.button(
        "← BACK TO SIGN IN",
        use_container_width=True,
    ):
        st.session_state.auth_view = "login"
        st.rerun()


def render_profile():
    """Show and persist edits to the current user's profile."""
    username = st.session_state.get("auth_username")
    user = get_user(username)

    if not user:
        return

    st.markdown(
        '<div class="section-title">My AURA AI Profile</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">USERNAME</div>
            <div class="metric-value">@{username}</div>
            <div class="metric-small">
                {user.get("role", "User")} account
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("profile_form"):
        display_name = st.text_input(
            "Full Name",
            value=user.get("display_name", ""),
        )

        email = st.text_input(
            "Email Address",
            value=user.get("email", ""),
        )

        phone = st.text_input(
            "Phone",
            value=user.get("phone", ""),
        )

        save = st.form_submit_button(
            "SAVE PROFILE",
            use_container_width=True,
        )

    if save:
        success, message = update_user_profile(
            username,
            display_name,
            email,
            phone,
        )

        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)


def require_authentication():
    init_auth_state()

    if not st.session_state.authenticated:
        render_login_page()
        return False

    if not get_user(st.session_state.get("auth_username")):
        logout()
        render_login_page()
        return False

    return True


# COMPANY LOGOS
# ============================================================

COMPANY_LOGO_DOMAINS = {
    "TCS": "tcs.com",
    "INFY": "infosys.com",
    "RELIANCE": "ril.com",
    "HDFCBANK": "hdfcbank.com",
    "ICICIBANK": "icicibank.com",
    "SBIN": "sbi.co.in",
    "ITC": "itcportal.com",
    "WIPRO": "wipro.com",
    "HCLTECH": "hcltech.com",
    "LT": "larsentoubro.com",
    "AAPL": "apple.com",
    "MSFT": "microsoft.com",
    "GOOGL": "google.com",
    "GOOG": "google.com",
    "AMZN": "amazon.com",
    "TSLA": "tesla.com",
    "NVDA": "nvidia.com",
    "META": "meta.com",
    "NFLX": "netflix.com",
    "ORCL": "oracle.com",
    "IBM": "ibm.com",
    "ADBE": "adobe.com",
    "INTC": "intel.com",
    "AMD": "amd.com",
    "CSCO": "cisco.com",
    "QCOM": "qualcomm.com",
    "AVGO": "broadcom.com",
    "CRM": "salesforce.com",
    "PYPL": "paypal.com",
    "UBER": "uber.com",
    "JPM": "jpmorganchase.com",
    "V": "visa.com",
    "MA": "mastercard.com",
    "KO": "coca-cola.com",
    "PEP": "pepsico.com",
    "MCD": "mcdonalds.com",
    "WMT": "walmart.com",
}

def get_company_logo_url(stock_name, ticker):
    ticker_key = str(ticker or "").upper().split(".")[0].strip()
    name_key = str(stock_name or "").upper().strip()
    domain = COMPANY_LOGO_DOMAINS.get(ticker_key) or COMPANY_LOGO_DOMAINS.get(name_key)

    if domain:
        return f"https://logos.hunter.io/{domain}"
    return None

def get_company_initials(stock_name, ticker):
    value = str(ticker or stock_name or "AI").upper().split(".")[0].strip()
    if len(value) >= 2:
        return value[:4]
    words = str(stock_name or "").split()
    if len(words) >= 2:
        return "".join(word[0] for word in words[:3]).upper()
    return value or "AI"

def company_logo_html(stock_name, ticker, small=False):
    logo_url = get_company_logo_url(stock_name, ticker)
    initials = get_company_initials(stock_name, ticker)

    if logo_url:
        css_class = "company-logo-small" if small else "company-logo"
        return (
            f'<img class="{css_class}" src="{logo_url}" '
            f'alt="{stock_name} logo" loading="eager">'
        )

    return f'<div class="company-logo-fallback">{initials}</div>'



# ============================================================
# AUTHENTICATION GATE
# ============================================================

# ==========================================================
# AURA AI — FINAL POLISHED APPLICATION UI
# ==========================================================

st.markdown(r"""
<style>
.aura-header-wrap { padding-top:4px; }

.aura-header-actions {
    display:flex!important;
    align-items:center!important;
    justify-content:flex-end!important;
    width:100%!important;
    gap:6px!important;
}

/* ==========================================================
   AURA HEADER ICONS — RELIABLE FIX
   Each icon is wrapped in st.container(key="...") in the Python
   code below, which gives Streamlit a REAL wrapping element with
   a stable class: .st-key-<name>. That element genuinely contains
   the popover, unlike separate st.markdown() marker spans, so
   these selectors reliably match.
   ========================================================== */

/* Remove every native chevron/caret Streamlit adds to popover buttons */
.st-key-aura_engine_wrap [data-testid="stPopover"] button svg,
.st-key-aura_bell_wrap [data-testid="stPopover"] button svg,
.st-key-aura_sun_wrap [data-testid="stPopover"] button svg,
.st-key-aura_profile_wrap [data-testid="stPopover"] button svg,
.st-key-aura_engine_wrap [data-testid="stPopover"] button [data-testid="stIconMaterial"],
.st-key-aura_bell_wrap [data-testid="stPopover"] button [data-testid="stIconMaterial"],
.st-key-aura_sun_wrap [data-testid="stPopover"] button [data-testid="stIconMaterial"],
.st-key-aura_profile_wrap [data-testid="stPopover"] button [data-testid="stIconMaterial"] {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
}

/* AI ENGINE ONLINE — pill, unchanged look */
.st-key-aura_engine_wrap [data-testid="stPopover"] button {
    width: 142px !important;
    min-width: 142px !important;
    height: 42px !important;
    min-height: 42px !important;
    padding: 0 14px !important;
    border: 1px solid rgba(34,211,238,.20) !important;
    border-radius: 999px !important;
    background: rgba(7,15,24,.58) !important;
    color: #67e8f9 !important;
    font-size: .68rem !important;
    font-weight: 750 !important;
    letter-spacing: .03em !important;
}
.st-key-aura_engine_wrap [data-testid="stPopover"] button:hover {
    border-color: rgba(34,211,238,.45) !important;
    background: rgba(11,28,40,.85) !important;
}

/* BELL — icon only, no box, no border, no arrow */
.st-key-aura_bell_wrap [data-testid="stPopover"] button {
    width: 42px !important;
    min-width: 42px !important;
    height: 42px !important;
    min-height: 42px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    outline: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #e8eef7 !important;
    font-size: 21px !important;
    font-weight: 400 !important;
}
.st-key-aura_bell_wrap [data-testid="stPopover"] button:hover {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #67e8f9 !important;
    transform: scale(1.08);
}

/* SUN / APPEARANCE — icon only, no box, no arrow */
.st-key-aura_sun_wrap [data-testid="stPopover"] button {
    width: 42px !important;
    min-width: 42px !important;
    height: 42px !important;
    min-height: 42px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    outline: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #e8eef7 !important;
    font-size: 22px !important;
    font-weight: 400 !important;
}
.st-key-aura_sun_wrap [data-testid="stPopover"] button:hover {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #67e8f9 !important;
    transform: scale(1.08);
}

/* PROFILE — perfect circle with initial + green online dot, no arrow */
.st-key-aura_profile_wrap [data-testid="stPopover"] button {
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 46px !important;
    min-width: 46px !important;
    max-width: 46px !important;
    height: 46px !important;
    min-height: 46px !important;
    max-height: 46px !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 50% !important;
    border: 1px solid rgba(255,255,255,.18) !important;
    background: #263143 !important;
    color: #f4f7fb !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    line-height: 1 !important;
    box-shadow: 0 4px 16px rgba(0,0,0,.28) !important;
}
.st-key-aura_profile_wrap [data-testid="stPopover"] button:hover {
    background: #303b50 !important;
    border-color: rgba(34,211,238,.35) !important;
    box-shadow: 0 0 22px rgba(34,211,238,.08) !important;
}
.st-key-aura_profile_wrap [data-testid="stPopover"] button::after {
    content: "" !important;
    position: absolute !important;
    width: 10px !important;
    height: 10px !important;
    right: 1px !important;
    bottom: 1px !important;
    background: #22c55e !important;
    border: 2px solid #07111c !important;
    border-radius: 50% !important;
    display: block !important;
}

/* Popover panels */
[data-testid="stPopoverBody"] {
    background:#0b111b!important;
    border:1px solid rgba(148,163,184,.14)!important;
    border-radius:14px!important;
    box-shadow:0 24px 60px rgba(0,0,0,.45)!important;
}
.aura-profile-panel-title {
    color:#f8fafc;
    font-size:1.05rem;
    font-weight:850;
    letter-spacing:.01em;
    margin-bottom:2px;
}
.aura-profile-panel-subtitle {
    color:#64748b;
    font-size:.68rem;
}
.aura-profile-avatar-large {
    width:70px;
    height:70px;
    margin:0 auto 12px;
    border-radius:50%;
    display:flex;
    align-items:center;
    justify-content:center;
    background:#263143;
    border:1px solid rgba(255,255,255,.18);
    color:#fff;
    font-size:22px;
    font-weight:850;
    position:relative;
}
.aura-profile-avatar-large::after {
    content:"";
    position:absolute;
    width:12px;
    height:12px;
    right:1px;
    bottom:1px;
    border-radius:50%;
    background:#22c55e;
    border:2px solid #0b111b;
}
.aura-setting-row {
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:12px 2px;
    border-bottom:1px solid rgba(148,163,184,.08);
}
.aura-setting-label { color:#dce5ef; font-size:.78rem; font-weight:650; }
.aura-setting-value { color:#718096; font-size:.72rem; }

@media (max-width:900px) {
    .st-key-aura_engine_wrap [data-testid="stPopover"] button {
        width:120px!important;
        min-width:120px!important;
        font-size:.61rem!important;
    }
    .st-key-aura_profile_wrap [data-testid="stPopover"] button {
        width:42px!important;
        height:42px!important;
        min-width:42px!important;
        min-height:42px!important;
    }
}

.hero-final { position:relative; overflow:hidden; margin:10px 0 24px; padding:28px 30px; min-height:205px; border-radius:18px; border:1px solid rgba(148,163,184,.12); background:radial-gradient(circle at 88% 45%,rgba(34,211,238,.10),transparent 30%),radial-gradient(circle at 72% 120%,rgba(59,130,246,.08),transparent 35%),linear-gradient(135deg,#0e1623,#080d15 72%); box-shadow:0 22px 55px rgba(0,0,0,.28); }
.hero-final::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:linear-gradient(180deg,#22d3ee,#3b82f6,#5eead4); }
.hero-grid { display:grid; grid-template-columns:minmax(0,1fr) 330px; gap:25px; align-items:center; }
.hero-label-final { color:#5eead4; font-size:.67rem; font-weight:850; letter-spacing:.14em; text-transform:uppercase; }
.hero-main { display:flex; align-items:center; gap:17px; margin-top:12px; }
.hero-main .company-logo,.hero-main .company-logo-fallback { width:68px;height:68px;min-width:68px; }
.hero-title-final { color:#f8fafc; font-size:1.72rem; font-weight:850; letter-spacing:-.025em; }
.hero-ticker-final { margin-top:8px; color:#718096; font-size:.78rem; font-weight:600; }
.hero-description { margin-top:18px; max-width:670px; color:#7f8da1; font-size:.77rem; line-height:1.65; }
.hero-chip-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:13px; }
.hero-chip { padding:6px 9px; border-radius:999px; color:#a9dfe7; background:rgba(34,211,238,.06); border:1px solid rgba(34,211,238,.14); font-size:.62rem; font-weight:750; letter-spacing:.04em; }
.hero-visual { height:155px; border:1px solid rgba(34,211,238,.10); border-radius:15px; background:rgba(5,11,19,.42); position:relative; overflow:hidden; }
.hero-visual-label { position:absolute; left:14px; top:11px; color:#536579; font-size:.58rem; letter-spacing:.12em; font-weight:800; }
.hero-visual svg { width:100%; height:100%; display:block; }
.ready-card { padding:30px; text-align:center; border-radius:16px; background:linear-gradient(145deg,#0e1521,#090e16); border:1px solid rgba(34,211,238,.12); box-shadow:0 15px 35px rgba(0,0,0,.22); }
.ready-orb { width:58px;height:58px;margin:0 auto 14px;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#67e8f9;font-weight:900;background:radial-gradient(circle,#14334a 0%,#0c1725 65%);border:1px solid rgba(34,211,238,.28);box-shadow:0 0 28px rgba(34,211,238,.10); }
.ready-title { color:#f5f7fa; font-size:1.12rem; font-weight:800; }
.ready-text { margin-top:7px; color:#65758b; font-size:.74rem; line-height:1.6; }
.chart-note { margin-top:-8px; margin-bottom:10px; color:#596b80; font-size:.66rem; }
.final-footer { margin-top:48px; padding:28px 12px 10px; text-align:center; color:#536176; font-size:.68rem; line-height:1.7; border-top:1px solid rgba(148,163,184,.08); }
.final-footer strong { color:#9eacbd; }
@media (max-width:900px) { .hero-grid{grid-template-columns:1fr;} .hero-visual{display:none;} .hero-final{padding:22px;} .hero-title-final{font-size:1.4rem;} .hero-description{max-width:none;} }

/* =========================================================
   AURA AI — SIDEBAR BRAND + NAVIGATION
   ========================================================= */
.sidebar-brand-row { display:flex; align-items:center; gap:12px; margin-bottom:22px; }
.sidebar-brand-icon { width:46px; height:46px; min-width:46px; border-radius:13px; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#0d3a4a,#0a1622); border:1px solid rgba(34,211,238,.35); color:#67e8f9; font-weight:900; font-size:1.05rem; box-shadow:0 0 22px rgba(34,211,238,.12); }
.sidebar-brand-text .brand { font-size:1.14rem; margin-bottom:0; }
.sidebar-brand-text .brand-subtitle { font-size:.54rem; margin-top:3px; line-height:1.4; }
.sidebar-nav-title, .sidebar-market-title { color:#68778c; font-size:.64rem; font-weight:800; letter-spacing:1.4px; margin:4px 0 10px 2px; text-transform:uppercase; }
.sidebar-market-title { margin-top:22px; }

div[data-testid="stSidebar"] [class*="st-key-nav_"] button {
    justify-content:flex-start !important;
    text-align:left !important;
    background:transparent !important;
    border:1px solid transparent !important;
    color:#9aa8bb !important;
    font-weight:650 !important;
    letter-spacing:.02em !important;
    box-shadow:none !important;
    padding-left:14px !important;
    min-height:40px !important;
    border-radius:10px !important;
}
div[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover {
    background:rgba(34,211,238,.06) !important;
    color:#e8eef7 !important;
}

.status-card { padding:14px 16px; margin-top:16px; background:linear-gradient(145deg,#0e141f,#090d15); border:1px solid rgba(148,163,184,0.11); border-radius:12px; }
.status-row { display:flex; align-items:center; justify-content:space-between; padding:5px 0; font-size:.72rem; }
.status-row .dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#22c55e; margin-right:6px; box-shadow:0 0 8px rgba(34,197,94,.6); }
.status-label { color:#e8eef7; font-weight:650; }
.status-value { color:#718096; }

/* =========================================================
   AURA AI — PAGE HERO
   ========================================================= */
.page-eyebrow { color:#5eead4; font-size:.68rem; font-weight:850; letter-spacing:.14em; text-transform:uppercase; }
.page-title { margin-top:6px; color:#f8fafc; font-size:1.9rem; font-weight:850; letter-spacing:-.02em; }
.page-subtitle { margin-top:6px; color:#7f8da1; font-size:.82rem; }

.stat-pill-row { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:22px; }
.stat-pill { padding:16px 18px; background:linear-gradient(145deg,#0e1521,#090e16); border:1px solid rgba(148,163,184,.11); border-radius:13px; }
.stat-pill-label { color:#68778c; font-size:.62rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
.stat-pill-value { margin-top:7px; color:#f8fafc; font-size:1.15rem; font-weight:800; }
.stat-pill-value.live-dot::before { content:"●"; color:#22c55e; margin-right:6px; font-size:.7rem; }
@media (max-width:900px) { .stat-pill-row { grid-template-columns:repeat(2,1fr); } }

/* =========================================================
   AURA AI — DASHBOARD CARDS
   ========================================================= */
.dash-card { padding:22px 24px; background:linear-gradient(145deg,#0e1521,#090e16); border:1px solid rgba(148,163,184,.11); border-radius:16px; box-shadow:0 14px 34px rgba(0,0,0,.22); margin-top:22px; }
.dash-card-title { color:#f8fafc; font-size:.95rem; font-weight:800; }
.dash-card-sub { margin-top:3px; color:#68778c; font-size:.72rem; }
.dash-price { margin-top:14px; color:#f8fafc; font-size:1.9rem; font-weight:850; }
.dash-price-change { margin-left:10px; font-size:.85rem; font-weight:750; }
.dash-price-change.up { color:#4ade80; }
.dash-price-change.down { color:#fb7185; }

.summary-row { display:flex; align-items:center; justify-content:space-between; padding:9px 0; border-bottom:1px solid rgba(148,163,184,.08); font-size:.78rem; gap:12px; }
.summary-row .k { color:#718096; white-space:nowrap; }
.summary-row .v { color:#e8eef7; font-weight:700; }

.empty-state { padding:26px; text-align:center; color:#65758b; font-size:.78rem; border:1px dashed rgba(148,163,184,.18); border-radius:12px; }

/* Timeframe tab buttons */
div[data-testid="stHorizontalBlock"] [class*="st-key-tf_"] button {
    min-height:32px !important;
    padding:0 10px !important;
    font-size:.68rem !important;
    font-weight:750 !important;
    border-radius:8px !important;
    background:transparent !important;
    border:1px solid rgba(148,163,184,.14) !important;
    color:#8b98ab !important;
    box-shadow:none !important;
}
</style>
""", unsafe_allow_html=True)

if not require_authentication():
    st.stop()

logged_in_username = st.session_state.get("auth_username")
logged_in_user = get_user(logged_in_username) if logged_in_username else None
profile_name = (logged_in_user or {}).get("display_name") or logged_in_username or "User"
profile_initial = profile_name.strip()[0].upper() if profile_name.strip() else "U"

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ==========================================================
# HEADER — CLEAN REFERENCE-STYLE ACTION BAR
# ==========================================================
header_left, header_actions = st.columns([7.7, 2.3], gap="small")

with header_left:
    st.markdown(
        f'<div class="aura-header-wrap">'
        f'<div class="brand" style="font-size:1.15rem;">AURA AI</div>'
        f'<div class="brand-subtitle">{html.escape(str(st.session_state.page)).upper()}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with header_actions:
    st.markdown('<div class="aura-header-actions">', unsafe_allow_html=True)
    engine_col, bell_col, sun_col, profile_col = st.columns(
        [3.25, 1, 1, 1.25], gap="small"
    )

    # ------------------------------------------------------
    # AI ENGINE ONLINE — clickable status pill, no arrow
    # ------------------------------------------------------
    with engine_col:
        with st.container(key="aura_engine_wrap"):
            with st.popover("●  AI ENGINE ONLINE", use_container_width=True):
                st.markdown("### AI Engine")
                st.success("System online")
                st.caption(
                    "AURA AI forecasting services are ready. "
                    "The selected stock can be analysed using its automatically selected trained model "
                    "and Monte Carlo uncertainty engine."
                )
                st.markdown("---")
                st.markdown(
                    f'<div class="aura-setting-row">'
                    f'<span class="aura-setting-label">Model</span>'
                    f'<span class="aura-setting-value">Auto Model + MC Dropout</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="aura-setting-row">'
                    f'<span class="aura-setting-label">User</span>'
                    f'<span class="aura-setting-value">{html.escape(profile_name)}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ------------------------------------------------------
    # NOTIFICATIONS — icon only, clicking opens the panel
    # ------------------------------------------------------
    with bell_col:
        with st.container(key="aura_bell_wrap"):
            with st.popover("🔔", use_container_width=True):
                st.markdown("### 🔔 Notifications")
                st.caption("AURA AI system updates")
                st.markdown("---")

                notifications = st.session_state.get("notifications", [])
                if not notifications:
                    st.info("No new notifications.")
                else:
                    for i, note in enumerate(notifications[:6]):
                        note_type = note.get("type", "UPDATE")
                        icon = {
                            "AI SIGNAL": "🟢",
                            "MARKET": "📈",
                            "RISK": "⚠️",
                            "SYSTEM": "◈",
                        }.get(note_type, "•")
                        st.markdown(f"**{icon} {note_type}**")
                        st.write(note.get("message", ""))
                        st.caption(note.get("time", "Now"))
                        if i < min(len(notifications), 6) - 1:
                            st.markdown("---")

    # ------------------------------------------------------
    # APPEARANCE — icon only, clicking opens a small panel
    # ------------------------------------------------------
    with sun_col:
        with st.container(key="aura_sun_wrap"):
            with st.popover("☼", use_container_width=True):
                st.markdown("### Appearance")
                st.caption("AURA AI terminal appearance")
                st.markdown("---")
                st.markdown(
                    '<div class="aura-setting-row">'
                    '<span class="aura-setting-label">Theme</span>'
                    '<span class="aura-setting-value">Dark Terminal</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.caption("The current dashboard uses the professional dark theme.")

    # ------------------------------------------------------
    # PROFILE — circular initial, clicking opens the profile panel
    # ------------------------------------------------------
    with profile_col:
        with st.container(key="aura_profile_wrap"):
            with st.popover(profile_initial, use_container_width=False):
                st.markdown(
                    f'<div style="text-align:center;padding:4px 0 16px;">'
                    f'<div class="aura-profile-avatar-large">'
                    f'{html.escape(profile_initial)}'
                    f'</div>'
                    f'<div class="aura-profile-panel-title">'
                    f'{html.escape(profile_name)}'
                    f'</div>'
                    f'<div class="aura-profile-panel-subtitle">'
                    f'@{html.escape(str(logged_in_username or "user"))}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("**PROFILE**")
                st.caption("Manage your AURA AI account details")

                with st.form("aura_final_profile_form", clear_on_submit=False):
                    display_name = st.text_input(
                        "Full Name",
                        value=logged_in_user.get("display_name", ""),
                    )
                    email = st.text_input(
                        "Email Address",
                        value=logged_in_user.get("email", ""),
                    )
                    phone = st.text_input(
                        "Phone",
                        value=logged_in_user.get("phone", ""),
                    )
                    save_profile = st.form_submit_button(
                        "SAVE PROFILE",
                        use_container_width=True,
                    )

                if save_profile:
                    success, message = update_user_profile(
                        logged_in_username,
                        display_name,
                        email,
                        phone,
                    )
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

                st.markdown("---")
                st.markdown("**SETTINGS**")
                st.markdown(
                    '<div class="aura-setting-row">'
                    '<span class="aura-setting-label">Theme</span>'
                    '<span class="aura-setting-value">Dark Terminal</span>'
                    '</div>'
                    '<div class="aura-setting-row">'
                    '<span class="aura-setting-label">Notifications</span>'
                    '<span class="aura-setting-value">Enabled</span>'
                    '</div>'
                    '<div class="aura-setting-row">'
                    '<span class="aura-setting-label">Account</span>'
                    f'<span class="aura-setting-value">{html.escape(str(logged_in_user.get("role", "User")))}</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                if st.button(
                    "LOG OUT",
                    use_container_width=True,
                    key="aura_final_logout",
                ):
                    logout()
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# SIDEBAR — BRAND + NAVIGATION + MARKET CONTROL
# ==========================================================
st.sidebar.markdown(
    '<div class="sidebar-brand-row">'
    '<div class="sidebar-brand-icon">◈</div>'
    '<div class="sidebar-brand-text">'
    '<div class="brand">AURA AI</div>'
    '<div class="brand-subtitle">ADAPTIVE UNCERTAINTY<br>RISK • ANALYTICS</div>'
    '</div></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="sidebar-nav-title">Navigate</div>', unsafe_allow_html=True)

NAV_ITEMS = [
    ("Dashboard", "▦"),
    ("Live Prediction", "⚡"),
    ("Uncertainty Map", "◈"),
    ("Monte Carlo", "🎲"),
    ("Backtesting", "📊"),
    ("Alerts", "🔔"),
]
for nav_label, nav_icon in NAV_ITEMS:
    nav_slug = nav_label.lower().replace(" ", "_")
    with st.sidebar.container(key=f"nav_{nav_slug}"):
        if st.button(f"{nav_icon}   {nav_label}", key=f"nav_{nav_slug}_btn", use_container_width=True):
            st.session_state.page = nav_label
            st.rerun()

active_nav_slug = st.session_state.page.lower().replace(" ", "_")
st.sidebar.markdown(
    f"<style>.st-key-nav_{active_nav_slug} button {{"
    f"background:rgba(34,211,238,.10) !important;"
    f"border-color:rgba(34,211,238,.35) !important;"
    f"color:#e8eef7 !important;}}</style>",
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="sidebar-market-title">Market</div>', unsafe_allow_html=True)
indian_stocks = [name for name, info in STOCKS.items() if info.get("market") == "India"]
us_stocks = [name for name, info in STOCKS.items() if info.get("market") == "USA"]
market = st.sidebar.radio("Market", ["🇮🇳 India", "🇺🇸 United States"], horizontal=False, label_visibility="collapsed")
available_stocks = indian_stocks if market.startswith("🇮🇳") else us_stocks
if not available_stocks:
    st.error("No stocks are configured for the selected market."); st.stop()
selected_stock = st.sidebar.selectbox("Stock", available_stocks, index=0)
config = STOCKS[selected_stock]
currency = config.get("currency", "₹")
ticker = config.get("ticker", "N/A")
company_name = config.get("name", selected_stock)
st.sidebar.divider()
st.sidebar.markdown(f'''<div class="info-panel"><div class="company-identity">{company_logo_html(selected_stock,ticker,small=True)}<div class="company-identity-text"><div class="company-identity-name">{html.escape(str(company_name))}</div><div class="company-identity-meta">{html.escape(str(ticker))} &nbsp;•&nbsp; {html.escape(str(config.get("market","")))}</div></div></div><div style="height:18px;"></div><div class="info-key">Market</div><div class="info-value">{get_market_flag(config.get("market"))} {html.escape(str(config.get("market","")))}</div><br><div class="info-key">AI Architecture</div><div class="info-value">LSTM + Monte Carlo Dropout</div></div>''', unsafe_allow_html=True)

stock_data = load_stock_data(selected_stock)

# ------------------------------------------------------------------
# AUTO-REFRESH: pull any missing recent rows via yfinance, once per
# session per stock, so "Data Through" doesn't silently go stale.
# ------------------------------------------------------------------
if "refreshed_stocks" not in st.session_state:
    st.session_state.refreshed_stocks = {}

last_data_date = get_last_data_date(stock_data)
stale = is_data_stale(last_data_date)
already_attempted_today = st.session_state.refreshed_stocks.get(selected_stock) == time.strftime("%Y-%m-%d")

refresh_clicked = st.sidebar.button("🔄  Refresh Data", key="manual_refresh_btn", use_container_width=True)

if refresh_clicked or (stale and not already_attempted_today):
    with st.sidebar:
        with st.spinner("Checking for newer market data..."):
            updated_df, refresh_error = refresh_stock_data(selected_stock, ticker, stock_data)
    st.session_state.refreshed_stocks[selected_stock] = time.strftime("%Y-%m-%d")
    if refresh_error:
        st.sidebar.warning(f"Auto-refresh failed: {refresh_error}")
    elif updated_df is not None and get_last_data_date(updated_df) != last_data_date:
        try:
            save_stock_data(selected_stock, updated_df)
            stock_data = updated_df
            st.sidebar.success("Data refreshed.")
        except Exception as exc:
            st.sidebar.warning(f"Fetched new data but could not save it: {exc}")
    elif refresh_clicked:
        st.sidebar.info("Already up to date.")

data_feed_state = "Live" if stock_data is not None else "Offline"
data_feed_color = "#22c55e" if stock_data is not None else "#f87171"
current_last_date = get_last_data_date(stock_data)
current_stale = is_data_stale(current_last_date)
data_through_display = current_last_date.strftime("%d %b %Y") if current_last_date is not None else "Unavailable"
data_through_color = "#f59e0b" if current_stale else "#718096"
st.sidebar.markdown(
    f'''<div class="status-card">
    <div class="status-row"><span class="status-label"><span class="dot"></span>Connected</span><span class="status-value">{time.strftime("%I:%M:%S %p")}</span></div>
    <div class="status-row"><span class="status-label">Data Feed</span><span class="status-value" style="color:{data_feed_color};">{data_feed_state}</span></div>
    <div class="status-row"><span class="status-label">Data Through</span><span class="status-value" style="color:{data_through_color};">{data_through_display}{" ⚠" if current_stale else ""}</span></div>
    </div>''',
    unsafe_allow_html=True,
)
st.sidebar.markdown('<div style="margin-top:16px;color:#536176;font-size:.68rem;line-height:1.6;">Predictions are generated from the selected stock\'s dedicated trained model.</div>', unsafe_allow_html=True)

# ==========================================================
# DATA STATUS + LIVE PREDICTION STATE (shared across pages)
# ==========================================================
if stock_data is not None:
    valid_dates = stock_data["Date"].dropna() if "Date" in stock_data.columns else pd.Series(dtype="datetime64[ns]")
    last_date = valid_dates.iloc[-1].strftime("%d %b %Y") if len(valid_dates) else "Unavailable"
    data_rows = len(stock_data)
else:
    last_date = "Unavailable"; data_rows = 0

if st.session_state.get("live_stock") != selected_stock:
    st.session_state.pop("live_result", None)
    st.session_state.pop("prediction_time", None)

live_result = st.session_state.get("live_result") if st.session_state.get("live_stock") == selected_stock else None


def run_live_prediction():
    """Run the real AURA AI prediction engine directly.

    The standalone backend has already been verified to return successfully.
    Running it directly here avoids the Windows subprocess/capture-output
    deadlock that can leave Streamlit stuck on the spinner.
    """
    try:
        st.session_state["live_result"] = None
        st.session_state["live_stock"] = selected_stock
        st.session_state["live_error"] = None

        # This is the same tested backend used by:
        # python -c "from src.live_prediction import generate_live_prediction; ..."
        with st.spinner(f"Running {selected_stock} AI prediction engine..."):
            result = generate_live_prediction(selected_stock)

        if not isinstance(result, dict):
            raise ValueError("Prediction engine returned an invalid result.")

        required = (
            "latest_price",
            "predicted_price",
            "percentage_change",
            "uncertainty",
        )
        missing = [key for key in required if key not in result]
        if missing:
            raise ValueError(
                "Prediction result is missing: " + ", ".join(missing)
            )

        # Normalize backend terminology to the dashboard terminology.
        signal = str(result.get("signal", "HOLD")).upper().strip()
        if signal in {"POSITIVE", "BULLISH", "BUY"}:
            result["signal"] = "BUY"
        elif signal in {"NEGATIVE", "BEARISH", "SELL"}:
            result["signal"] = "SELL"
        else:
            result["signal"] = "HOLD"

        # Make sure numeric values are JSON/UI safe before storing them.
        for key in (
            "latest_price",
            "predicted_price",
            "percentage_change",
            "uncertainty",
            "lower_bound",
            "upper_bound",
        ):
            if key in result:
                result[key] = float(result[key])

        result["status"] = "success"

        st.session_state["live_result"] = result
        st.session_state["live_stock"] = selected_stock
        st.session_state["prediction_time"] = time.strftime(
            "%d %b %Y • %H:%M:%S"
        )
        st.session_state["live_error"] = None

        return result

    except Exception as exc:
        st.session_state["live_result"] = None
        st.session_state["live_stock"] = selected_stock
        st.session_state["live_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return None


live_error = st.session_state.pop("live_error", None)
if live_error:
    st.error(f"AI prediction failed: {live_error}")

# Refresh state after a button click in the same Streamlit run.
live_result = st.session_state.get("live_result") if st.session_state.get("live_stock") == selected_stock else None

parsed = None
if live_result is not None:
    try:
        parsed = {}
        parsed["latest_price"] = float(live_result.get("latest_price", 0))
        parsed["predicted_price"] = float(live_result.get("predicted_price", 0))
        parsed["percentage_change"] = float(live_result.get("percentage_change", 0))
        parsed["uncertainty"] = abs(float(live_result.get("uncertainty", 0)))
        parsed["lower_bound"] = float(live_result.get("lower_bound", parsed["predicted_price"] - parsed["uncertainty"]))
        parsed["upper_bound"] = float(live_result.get("upper_bound", parsed["predicted_price"] + parsed["uncertainty"]))
        # Use the authoritative values returned by the live prediction engine.
        # Do not recalculate risk or confidence here.
        parsed["model"] = str(live_result.get("model", "Unknown"))
        parsed["model_path"] = str(live_result.get("model_path", ""))
        parsed["uncertainty_method"] = str(live_result.get("uncertainty_method", "Monte Carlo Dropout"))
        parsed["horizon"] = str(live_result.get("horizon", "1 Day"))
        parsed["sequence_length"] = int(live_result.get("sequence_length", 60))
        parsed["signal"] = str(live_result.get("signal", "HOLD")).upper()
        parsed["risk"] = str(live_result.get("risk", "UNKNOWN")).upper()
        parsed["confidence"] = float(live_result.get("confidence", 0.0))
        parsed["confidence_label"] = str(live_result.get("confidence_label", "95% prediction interval"))
        parsed["samples"] = live_result.get("samples", []) or []
        parsed["mc_samples"] = int(live_result.get("mc_samples", len(parsed["samples"])))
        parsed["data_through"] = str(live_result.get("data_through", last_date))
        parsed["rows_used"] = int(live_result.get("rows_used", data_rows))
        st.session_state["notifications"] = [
            {"type": "AI SIGNAL", "message": f"{selected_stock}: {parsed['signal']} signal generated.", "time": st.session_state.get("prediction_time", "Now")},
            {"type": "MARKET", "message": f"Forecast: {format_currency(parsed['predicted_price'], currency)} ({parsed['percentage_change']:+.2f}%).", "time": "Latest run"},
            {"type": "RISK", "message": f"Model risk is {parsed['risk']}; uncertainty ±{format_currency(parsed['uncertainty'], currency)}.", "time": "Latest run"},
        ]
    except (TypeError, ValueError) as exc:
        st.error(f"Prediction result contains invalid numeric data: {exc}")
        parsed = None

page = st.session_state.page

# ==========================================================
# PAGE: DASHBOARD
# ==========================================================
if page == "Dashboard":
    st.markdown('<div class="page-eyebrow">AURA AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Intelligent Stock Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Adaptive uncertainty • Risk aware • AI powered predictions</div>', unsafe_allow_html=True)

    confidence_display = f"{parsed['confidence']:.1f}%" if parsed else "—"
    status_display = "Live" if parsed else "Idle"
    st.markdown(
        f'''<div class="stat-pill-row">
        <div class="stat-pill"><div class="stat-pill-label">Model</div><div class="stat-pill-value">{html.escape(parsed["model"]) if parsed else "Auto-selected"}</div></div>
        <div class="stat-pill"><div class="stat-pill-label">Uncertainty</div><div class="stat-pill-value">Monte Carlo</div></div>
        <div class="stat-pill"><div class="stat-pill-label">Confidence</div><div class="stat-pill-value">{confidence_display}</div></div>
        <div class="stat-pill"><div class="stat-pill-label">Status</div><div class="stat-pill-value live-dot">{status_display}</div></div>
        </div>''',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.55, 1], gap="medium")

    with left_col:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="dash-card-title">Live Price Overview</div>'
            f'<div class="dash-card-sub">{html.escape(str(company_name))} • {html.escape(str(ticker))}</div>',
            unsafe_allow_html=True,
        )

        if "timeframe" not in st.session_state:
            st.session_state.timeframe = "3M"
        TIMEFRAMES = ["1W", "1M", "3M", "1Y", "All"]
        tf_cols = st.columns(len(TIMEFRAMES))
        for tf, tcol in zip(TIMEFRAMES, tf_cols):
            with tcol:
                with st.container(key=f"tf_{tf}"):
                    if st.button(tf, key=f"tf_{tf}_btn", use_container_width=True):
                        st.session_state.timeframe = tf
                        st.rerun()
        st.markdown(
            f"<style>.st-key-tf_{st.session_state.timeframe} button {{"
            f"background:rgba(34,211,238,.12) !important;"
            f"border-color:rgba(34,211,238,.4) !important;"
            f"color:#67e8f9 !important;}}</style>",
            unsafe_allow_html=True,
        )

        if stock_data is not None and "Close" in stock_data.columns and "Date" in stock_data.columns:
            df_clean = stock_data.dropna(subset=["Date", "Close"]).copy()
            days = get_timeframe_days(st.session_state.timeframe)
            plot_df = df_clean.tail(days) if days else df_clean
            if len(plot_df) >= 2:
                period_change = plot_df["Close"].iloc[-1] - plot_df["Close"].iloc[0]
                period_change_pct = (period_change / plot_df["Close"].iloc[0] * 100) if plot_df["Close"].iloc[0] else 0
                change_class = "up" if period_change >= 0 else "down"
                arrow = "▲" if period_change >= 0 else "▼"
                st.markdown(
                    f'<div class="dash-price">{format_currency(plot_df["Close"].iloc[-1], currency)}'
                    f'<span class="dash-price-change {change_class}">{arrow} {format_currency(abs(period_change), currency)} ({period_change_pct:+.2f}%)</span></div>',
                    unsafe_allow_html=True,
                )
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=plot_df["Date"], y=plot_df["Close"], mode="lines",
                    line=dict(width=2.2, color="#22d3ee"),
                    fill="tozeroy", fillcolor="rgba(34,211,238,0.06)",
                    hovertemplate="%{x|%d %b %Y}<br>" + currency + "%{y:,.2f}<extra></extra>",
                ))
                fig.update_layout(
                    height=340, margin=dict(l=10, r=10, t=16, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa8bb"),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickprefix=currency),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown('<div class="empty-state">Not enough data points for this timeframe.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state">No historical dataset found for this stock.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown('<div class="dash-card-title">Model Summary</div>', unsafe_allow_html=True)
        donut_fig = render_confidence_donut(parsed["confidence"] if parsed else 0.0)
        st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'''<div class="summary-row"><span class="k">Model</span><span class="v">{html.escape(parsed["model"]) if parsed else "Auto-selected"}</span></div>
            <div class="summary-row"><span class="k">Horizon</span><span class="v">{html.escape(parsed["horizon"]) if parsed else "1 Day"}</span></div>
            <div class="summary-row"><span class="k">Method</span><span class="v">{html.escape(parsed["uncertainty_method"]) if parsed else "Monte Carlo Dropout"}</span></div>
            <div class="summary-row"><span class="k">Status</span><span class="v">{"Live" if parsed else "Idle — run analysis"}</span></div>''',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown('<div class="dash-card-title">Prediction (Next Day)</div>', unsafe_allow_html=True)
        if parsed:
            arrow = "▲" if parsed["percentage_change"] >= 0 else "▼"
            change_class = "up" if parsed["percentage_change"] >= 0 else "down"
            st.markdown(
                f'<div class="dash-price" style="font-size:1.55rem;">{format_currency(parsed["predicted_price"], currency)}'
                f'<span class="dash-price-change {change_class}">{arrow} {parsed["percentage_change"]:+.2f}%</span></div>',
                unsafe_allow_html=True,
            )
            if stock_data is not None and "Close" in stock_data.columns:
                spark_values = list(stock_data["Close"].dropna().tail(30)) + [parsed["predicted_price"]]
                st.plotly_chart(render_sparkline(spark_values), use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(
                '<div class="empty-state">No forecast yet.<br>Run analysis on the Live Prediction page.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="dash-card-title">Uncertainty Range (95%)</div>'
        '<div class="dash-card-sub">Projected — widened using √t scaling from the latest Monte Carlo run. Not a stored multi-day backtest.</div>',
        unsafe_allow_html=True,
    )
    if parsed:
        import plotly.graph_objects as go
        horizon = list(range(1, 8))
        base_unc = parsed["uncertainty"]
        upper_path = [parsed["predicted_price"] + base_unc * (d ** 0.5) for d in horizon]
        lower_path = [parsed["predicted_price"] - base_unc * (d ** 0.5) for d in horizon]
        mid_path = [parsed["predicted_price"] for _ in horizon]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=horizon + horizon[::-1], y=upper_path + lower_path[::-1], fill="toself",
                                  fillcolor="rgba(34,211,238,0.10)", line=dict(width=0), hoverinfo="skip", name="95% Range"))
        fig.add_trace(go.Scatter(x=horizon, y=mid_path, mode="lines", line=dict(width=2, dash="dot", color="#67e8f9"), name="Prediction"))
        fig.add_trace(go.Scatter(x=horizon, y=upper_path, mode="lines", line=dict(width=1, color="#67e8f9"), name="Upper Bound"))
        fig.add_trace(go.Scatter(x=horizon, y=lower_path, mode="lines", line=dict(width=1, color="#67e8f9"), name="Lower Bound"))
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#9aa8bb"),
            xaxis=dict(title="Days ahead", showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickprefix=currency),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown('<div class="empty-state">Run AI analysis to see the projected uncertainty range.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# PAGE: LIVE PREDICTION
# ==========================================================
elif page == "Live Prediction":
    st.markdown('<div class="page-eyebrow">AURA AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Live Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Run the automatically selected trained model with Monte Carlo Dropout uncertainty on demand.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:26px;">Live AI Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    if st.button(
        f"⚡ RUN {str(selected_stock).upper()} AI ANALYSIS",
        use_container_width=True,
        key="run_final_prediction",
        type="primary",
    ):
        # The prediction is executed in a separate Python process and stored
        # immediately. No st.rerun() is needed; the current script continues
        # and renders the freshly stored result below.
        run_live_prediction()

    # Re-read the state after a completed prediction.
    live_result = st.session_state.get("live_result") if st.session_state.get("live_stock") == selected_stock else None
    if live_result is not None:
        try:
            parsed = {}
            parsed["latest_price"] = float(live_result.get("latest_price", 0))
            parsed["predicted_price"] = float(live_result.get("predicted_price", 0))
            parsed["percentage_change"] = float(live_result.get("percentage_change", 0))
            parsed["uncertainty"] = abs(float(live_result.get("uncertainty", 0)))
            parsed["lower_bound"] = float(live_result.get("lower_bound", parsed["predicted_price"] - parsed["uncertainty"]))
            parsed["upper_bound"] = float(live_result.get("upper_bound", parsed["predicted_price"] + parsed["uncertainty"]))
            parsed["model"] = str(live_result.get("model", "Unknown"))
            parsed["model_path"] = str(live_result.get("model_path", ""))
            parsed["uncertainty_method"] = str(live_result.get("uncertainty_method", "Monte Carlo Dropout"))
            parsed["horizon"] = str(live_result.get("horizon", "1 Day"))
            parsed["sequence_length"] = int(live_result.get("sequence_length", 60))
            parsed["signal"] = str(live_result.get("signal", "HOLD")).upper()
            parsed["risk"] = str(live_result.get("risk", "UNKNOWN")).upper()
            parsed["confidence"] = float(live_result.get("confidence", 0.0))
            parsed["confidence_label"] = str(live_result.get("confidence_label", "95% prediction interval"))
            parsed["samples"] = live_result.get("samples", []) or []
            parsed["mc_samples"] = int(live_result.get("mc_samples", len(parsed["samples"])))
            parsed["data_through"] = str(live_result.get("data_through", last_date))
            parsed["rows_used"] = int(live_result.get("rows_used", data_rows))
        except (TypeError, ValueError) as exc:
            st.error(f"Prediction result contains invalid numeric data: {exc}")
            parsed = None

    if parsed:
        cols = st.columns(4)
        metric_data = [
            ("Live Market Price", format_currency(parsed["latest_price"], currency), "Latest market observation"),
            ("AI Forecast", format_currency(parsed["predicted_price"], currency), "Next predicted price"),
            ("Expected Movement", ("▲ " if parsed["percentage_change"] >= 0 else "▼ ") + f"{parsed['percentage_change']:+.2f}%", "AI forecast vs current price"),
            ("AI Uncertainty", "±" + format_currency(parsed["uncertainty"], currency), "Monte Carlo prediction spread"),
        ]
        for col, (label, value, small) in zip(cols, metric_data):
            with col:
                st.markdown(f'''<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-small">{small}</div></div>''', unsafe_allow_html=True)

        st.markdown('<div class="section-title">AI Decision Layer</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
        signal_map = {
            "BUY": ("signal-positive", "▲ BUY"),
            "SELL": ("signal-negative", "▼ SELL"),
            "HOLD": ("signal-neutral", "● HOLD"),
        }
        signal_class, signal_text = signal_map.get(
            parsed["signal"].upper(),
            ("signal-neutral", f"● {parsed['signal'].upper()}")
        )
        risk_symbol = {"LOW": "◉", "MODERATE": "◐", "HIGH": "◑", "VERY HIGH": "◉"}.get(parsed["risk"], "●")
        d1, d2, d3 = st.columns([1.1, 1, 1.5])
        with d1:
            st.markdown(f'''<div class="signal-card"><div class="metric-label">AI SIGNAL</div><div class="{signal_class}">{signal_text}</div><div class="metric-small">Based on model forecast</div></div>''', unsafe_allow_html=True)
        with d2:
            st.markdown(f'''<div class="signal-card"><div class="metric-label">MODEL RISK</div><div style="color:#f5f7fa;font-size:1.35rem;font-weight:800;margin-top:9px;">{risk_symbol} {parsed["risk"]}</div><div class="metric-small">{risk_message(parsed["risk"])}</div></div>''', unsafe_allow_html=True)
        with d3:
            st.markdown(f'''<div class="signal-card"><div class="metric-label">95% PREDICTION RANGE</div><div style="color:#f5f7fa;font-size:1.18rem;font-weight:800;margin-top:9px;">{format_currency(parsed["lower_bound"], currency)} &nbsp;|&nbsp; {format_currency(parsed["upper_bound"], currency)}</div><div class="metric-small">Uncertainty interval produced by the AI engine</div></div>''', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Market Trajectory</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-note">Historical price • AI forecast • shaded 95% prediction interval</div>', unsafe_allow_html=True)
        try:
            import plotly.graph_objects as go
            if stock_data is not None and "Close" in stock_data.columns and "Date" in stock_data.columns:
                chart_df = stock_data.dropna(subset=["Date", "Close"]).tail(180).copy()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=chart_df["Date"], y=chart_df["Close"], mode="lines", name="Historical Price", line=dict(width=2.5)))
                if len(chart_df):
                    last_date_obj = chart_df["Date"].iloc[-1]
                    future_date = last_date_obj + pd.Timedelta(days=1)
                    fig.add_trace(go.Scatter(x=[last_date_obj, future_date], y=[parsed["latest_price"], parsed["predicted_price"]], mode="lines+markers", name="AI Forecast", line=dict(width=3, dash="dot"), marker=dict(size=9)))
                    fig.add_trace(go.Scatter(x=[last_date_obj, future_date, future_date, last_date_obj], y=[parsed["latest_price"], parsed["upper_bound"], parsed["lower_bound"], parsed["latest_price"]], fill="toself", fillcolor="rgba(34,211,238,0.10)", line=dict(width=0), hoverinfo="skip", name="95% Range"))
                    fig.add_trace(go.Scatter(x=[future_date], y=[parsed["upper_bound"]], mode="markers", name="Upper Bound", marker=dict(size=7)))
                    fig.add_trace(go.Scatter(x=[future_date], y=[parsed["lower_bound"]], mode="markers", name="Lower Bound", marker=dict(size=7)))
                fig.update_layout(height=470, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#9aa8bb"), hovermode="x unified", xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickprefix=currency), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Historical chart data is unavailable.")
        except ImportError:
            st.warning("Plotly is required for the advanced chart.")
        except Exception as exc:
            st.warning(f"Unable to render the forecast chart: {exc}")

        st.markdown('<div class="section-title">Model Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
        info = [
            ("MODEL", parsed["model"], "Automatically selected best model"),
            ("UNCERTAINTY", parsed["uncertainty_method"], f"{parsed['mc_samples']} stochastic samples"),
            ("DATA POINTS", f"{parsed['rows_used']:,}", "Observations used by prediction engine"),
            ("DATA THROUGH", parsed["data_through"], "Latest data used by prediction engine"),
        ]
        info_cols = st.columns(4)
        for col, (label, value, small) in zip(info_cols, info):
            with col:
                st.markdown(f'''<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:1.05rem;">{value}</div><div class="metric-small">{small}</div></div>''', unsafe_allow_html=True)
        if st.session_state.get("prediction_time"):
            st.caption(f"Last AI analysis: {st.session_state['prediction_time']} • Selected model: {ticker}")
    else:
        st.markdown('''<div class="ready-card"><div class="ready-orb">AI</div><div class="ready-title">AI Engine Ready</div><div class="ready-text">Run AI analysis to generate the next-price forecast, Monte Carlo uncertainty estimate, 95% prediction range and model risk assessment.</div></div>''', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Historical Market Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    if stock_data is not None:
        chart = create_historical_chart(stock_data)
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)
        with st.expander(f"View {selected_stock} raw market data"):
            st.dataframe(stock_data.tail(30), use_container_width=True, hide_index=True)
    else:
        st.warning(f"No historical dataset found for {selected_stock}.")

# ==========================================================
# PAGE: UNCERTAINTY MAP
# ==========================================================
elif page == "Uncertainty Map":
    st.markdown('<div class="page-eyebrow">AURA AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Uncertainty Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">How the model\'s confidence interval widens further into the future.</div>', unsafe_allow_html=True)

    if parsed:
        import plotly.graph_objects as go
        horizon = list(range(1, 15))
        base_unc = parsed["uncertainty"]
        upper_path = [parsed["predicted_price"] + base_unc * (d ** 0.5) for d in horizon]
        lower_path = [parsed["predicted_price"] - base_unc * (d ** 0.5) for d in horizon]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=horizon + horizon[::-1], y=upper_path + lower_path[::-1], fill="toself", fillcolor="rgba(34,211,238,0.10)", line=dict(width=0), hoverinfo="skip", name="Uncertainty Band"))
        fig.add_trace(go.Scatter(x=horizon, y=[parsed["predicted_price"]] * len(horizon), mode="lines", line=dict(dash="dot", color="#67e8f9"), name="Prediction"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#9aa8bb"), xaxis=dict(title="Days ahead", showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickprefix=currency))
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        st.markdown('<div class="dash-card-title">Projected Confidence Cone</div><div class="dash-card-sub">√t scaling from the latest Monte Carlo uncertainty — a projection, not a stored multi-day backtest.</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">Run AI analysis on the Live Prediction page first.</div>', unsafe_allow_html=True)

    backtest_path = find_predictions_file(selected_stock)
    metrics_path = find_metrics_file(selected_stock)

    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.markdown('<div class="dash-card-title">Prediction vs Actual</div>', unsafe_allow_html=True)

    bt_df = load_backtest_df(backtest_path)
    if bt_df is not None and {"Actual", "Predicted"}.issubset(bt_df.columns):
        try:
            import plotly.graph_objects as go
            x_axis = bt_df["Date"] if "Date" in bt_df.columns else bt_df.index
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_axis, y=bt_df["Actual"], mode="lines", name="Actual Price", line=dict(width=2.2)))
            fig.add_trace(go.Scatter(x=x_axis, y=bt_df["Predicted"], mode="lines", name="Predicted Price", line=dict(width=2, dash="dot")))
            fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa8bb"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickprefix=currency),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.markdown(f'<div class="empty-state">Could not read backtest file: {html.escape(str(exc))}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="empty-state">No backtest results found yet.<br>'
            f'Expected <code>results/{selected_stock.lower().replace(" ", "_")}_predictions.npz</code> '
            f'with keys <code>actual</code>, <code>predicted</code>.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
# ==========================================================
# PAGE: BACKTESTING
# ==========================================================
elif page == "Backtesting":
    st.markdown('<div class="page-eyebrow">AURA AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Backtesting</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Historical accuracy of the trained model against actual prices.</div>', unsafe_allow_html=True)

    backtest_path = find_predictions_file(selected_stock)
    metrics_path = find_metrics_file(selected_stock)

    backtest_path = find_predictions_file(selected_stock)
    metrics_path = find_metrics_file(selected_stock)

    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.markdown('<div class="dash-card-title">Prediction vs Actual</div>', unsafe_allow_html=True)

    bt_df = load_backtest_df(backtest_path)   # <-- must use this, NOT pd.read_csv

    if bt_df is not None and {"Actual", "Predicted"}.issubset(bt_df.columns):
        try:
            import plotly.graph_objects as go
            x_axis = bt_df["Date"] if "Date" in bt_df.columns else bt_df.index
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_axis, y=bt_df["Actual"], mode="lines", name="Actual Price", line=dict(width=2.2)))
            fig.add_trace(go.Scatter(x=x_axis, y=bt_df["Predicted"], mode="lines", name="Predicted Price", line=dict(width=2, dash="dot")))
            fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa8bb"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickprefix=currency),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.markdown(
                f'<div class="empty-state">Could not read backtest file: {html.escape(str(exc))}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="empty-state">No backtest results found yet.<br>'
            f'Expected <code>results/{selected_stock.lower().replace(" ", "_")}_predictions.npz</code>.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    st.markdown('<div class="dash-card-title">Performance Metrics</div>', unsafe_allow_html=True)
    if metrics_path is not None:
        try:
            if metrics_path.suffix == ".json":
                import json
                with open(metrics_path) as f:
                    metrics = json.load(f)
            else:
                metrics_df = pd.read_csv(metrics_path)
                metrics = dict(zip(metrics_df.iloc[:, 0], metrics_df.iloc[:, 1])) if metrics_df.shape[1] >= 2 else {}
            mcols = st.columns(4)
            labels = ["MAE", "RMSE", "MAPE", "R2"]
            for mcol, key in zip(mcols, labels):
                val = metrics.get(key) if isinstance(metrics, dict) else None
                if val is None and isinstance(metrics, dict):
                    val = metrics.get(key.lower())
                with mcol:
                    st.markdown(f'''<div class="metric-card"><div class="metric-label">{key}</div><div class="metric-value">{val if val is not None else "—"}</div></div>''', unsafe_allow_html=True)
        except Exception as exc:
            st.markdown(f'<div class="empty-state">Could not read metrics file: {html.escape(str(exc))}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">No saved metrics file found for this stock yet.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# PAGE: ALERTS
# ==========================================================
else:
    st.markdown('<div class="page-eyebrow">AURA AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Alerts</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">System and market notifications from your latest AI runs.</div>', unsafe_allow_html=True)

    notifications = st.session_state.get("notifications", [])
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    if not notifications:
        st.markdown('<div class="empty-state">No notifications yet. Run AI analysis to generate alerts.</div>', unsafe_allow_html=True)
    else:
        for note in notifications:
            note_type = note.get("type", "UPDATE")
            icon = {"AI SIGNAL": "🟢", "MARKET": "📈", "RISK": "⚠️", "SYSTEM": "◈"}.get(note_type, "•")
            st.markdown(
                f'<div class="summary-row"><span class="k">{icon} {note_type}</span>'
                f'<span class="v" style="font-weight:600;text-align:right;">{html.escape(str(note.get("message","")))}</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('''<div class="final-footer"><strong>AURA AI — INTELLIGENT STOCK ANALYTICS</strong><br>Multi-Model Forecasting • Monte Carlo Dropout • Uncertainty-Aware Prediction<br><br>Research &amp; Educational Use • Not Financial Advice</div>''', unsafe_allow_html=True)