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
/* ================================
   TOP RIGHT PROFILE BUTTON
   ================================ */

div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) div[data-testid="stPopover"] > button {
    width: 58px !important;
    height: 58px !important;
    min-width: 58px !important;
    min-height: 58px !important;

    padding: 0 !important;

    border-radius: 50% !important;

    background: #263143 !important;

    border: 1px solid rgba(255,255,255,0.20) !important;

    color: #f4f7fb !important;

    font-size: 18px !important;
    font-weight: 700 !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    position: relative !important;

    box-shadow: 0 4px 15px rgba(0,0,0,0.30) !important;
}

/* Online green dot */

div[data-testid="stPopover"] > button::after {
    content: "";

    position: absolute;

    width: 11px;
    height: 11px;

    right: 1px;
    bottom: 1px;

    background: #22c55e;

    border: 2px solid #07111c;

    border-radius: 50%;
}

/* =====================================================
   AURA AI - NOTIFICATION BELL
   ===================================================== */

/* Bell only — no circle */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) [data-testid="stPopover"] > button {
    width: auto !important;
    height: auto !important;
    min-width: 0 !important;
    min-height: 0 !important;
    max-width: none !important;
    max-height: none !important;

    padding: 0 !important;
    margin: 0 !important;

    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;

    transform: none !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Bell itself */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) [data-testid="stPopover"] > button p {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 20px !important;
    line-height: 1 !important;
}

/* Bell hover */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) [data-testid="stPopover"] > button:hover {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
/* ==========================================================
   AURA AI — FINAL HEADER FIX
   Force Streamlit popovers to look like pure custom controls
   ========================================================== */

/* ---------- REMOVE STREAMLIT POPOVER ARROWS ---------- */

div[data-testid="stPopover"] > button::after {
    display: none !important;
    content: none !important;
}

div[data-testid="stPopover"] > button svg,
div[data-testid="stPopover"] > button [data-testid="stIconMaterial"],
div[data-testid="stPopover"] > button [data-testid="stIcon"],
div[data-testid="stPopover"] > button span[aria-hidden="true"] {
    display: none !important;
    visibility: hidden !important;
}


/* ==========================================================
   AI ENGINE ONLINE
   ========================================================== */

div[data-testid="column"]:has(.aura-engine-marker)
div[data-testid="stPopover"] > button {
    width: 142px !important;
    min-width: 142px !important;
    height: 42px !important;
    min-height: 42px !important;

    padding: 0 !important;
    margin: 0 !important;

    border: 1px solid rgba(34,211,238,.22) !important;
    border-radius: 999px !important;

    background: rgba(7,15,24,.65) !important;
    box-shadow: none !important;

    color: #67e8f9 !important;
    font-size: 0 !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Re-create the visible text ourselves */
div[data-testid="column"]:has(.aura-engine-marker)
div[data-testid="stPopover"] > button::before {
    content: "●  AI ENGINE ONLINE" !important;

    display: block !important;

    color: #67e8f9 !important;
    font-family: "Inter", "Segoe UI", sans-serif !important;
    font-size: 10px !important;
    font-weight: 750 !important;
    letter-spacing: .03em !important;
}

div[data-testid="column"]:has(.aura-engine-marker)
div[data-testid="stPopover"] > button:hover {
    border-color: rgba(34,211,238,.50) !important;
    background: rgba(11,28,40,.85) !important;
}


/* ==========================================================
   BELL
   NO BOX
   NO BORDER
   NO ARROW
   ========================================================== */

div[data-testid="column"]:has(.aura-bell-marker)
div[data-testid="stPopover"] > button {
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

    font-size: 0 !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Create the bell ourselves */
div[data-testid="column"]:has(.aura-bell-marker)
div[data-testid="stPopover"] > button::before {
    content: "🔔" !important;

    display: block !important;

    font-size: 20px !important;
    line-height: 1 !important;

    filter: grayscale(1) brightness(1.8);
}

div[data-testid="column"]:has(.aura-bell-marker)
div[data-testid="stPopover"] > button:hover {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    transform: scale(1.08) !important;
}


/* ==========================================================
   APPEARANCE / SUN
   NO BOX
   NO ARROW
   ========================================================== */

div[data-testid="column"]:has(.aura-sun-marker)
div[data-testid="stPopover"] > button {
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

    font-size: 0 !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[data-testid="column"]:has(.aura-sun-marker)
div[data-testid="stPopover"] > button::before {
    content: "☼" !important;

    display: block !important;

    color: #e8eef7 !important;
    font-size: 23px !important;
    line-height: 1 !important;
}

div[data-testid="column"]:has(.aura-sun-marker)
div[data-testid="stPopover"] > button:hover {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    transform: scale(1.08) !important;
}


/* ==========================================================
   PROFILE
   CIRCLE ONLY
   H INSIDE
   GREEN ONLINE DOT
   NO ARROW
   ========================================================== */

div[data-testid="column"]:has(.aura-profile-marker)
div[data-testid="stPopover"] > button {
    position: relative !important;

    width: 58px !important;
    min-width: 58px !important;
    max-width: 58px !important;

    height: 58px !important;
    min-height: 58px !important;
    max-height: 58px !important;

    padding: 0 !important;
    margin: 0 !important;

    border-radius: 50% !important;

    border: 1px solid rgba(255,255,255,.20) !important;

    background: #263143 !important;

    box-shadow: 0 4px 16px rgba(0,0,0,.30) !important;

    font-size: 0 !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Re-create profile initial */
div[data-testid="column"]:has(.aura-profile-marker)
div[data-testid="stPopover"] > button::before {
    content: "H" !important;

    display: block !important;

    color: #f4f7fb !important;

    font-family: "Inter", "Segoe UI", sans-serif !important;
    font-size: 18px !important;
    font-weight: 800 !important;

    line-height: 1 !important;
}

/* Green online dot */
div[data-testid="column"]:has(.aura-profile-marker)
div[data-testid="stPopover"] > button::after {
    content: "" !important;

    display: block !important;

    position: absolute !important;

    width: 10px !important;
    height: 10px !important;

    right: 1px !important;
    bottom: 1px !important;

    background: #22c55e !important;

    border: 2px solid #07111c !important;

    border-radius: 50% !important;
}

/* Profile hover */
div[data-testid="column"]:has(.aura-profile-marker)
div[data-testid="stPopover"] > button:hover {
    background: #303b50 !important;

    border-color: rgba(34,211,238,.35) !important;

    box-shadow: 0 0 22px rgba(34,211,238,.10) !important;
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




# ============================================================
# ============================================================
# AUTHENTICATION & USER PROFILES
# ============================================================

# Persistent local SQLite user store.
# This fixes the previous problem where AUTH_USERS was only an
# in-memory dictionary and all newly-created users disappeared
# whenever Streamlit restarted/reran the app.
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

        # Keep the original demo accounts available.
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
    """Secure PBKDF2 password hash for the local prototype."""
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
    """Verify PBKDF2 hashes and legacy SHA-256 demo hashes."""
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

    # Legacy SHA-256 demo accounts.
    candidate = hashlib.sha256(
        str(password).encode("utf-8")
    ).hexdigest()

    return hmac.compare_digest(candidate, str(stored_hash))


def get_user(username):
    """Return a user dictionary from the persistent database."""
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
    """Create a persistent local user profile."""
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
    """Persist profile changes for the logged-in user."""
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
    """Reset a password after matching username and registered email."""
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
        """
        <div class="auth-card">
            <div class="auth-logo">AI</div>
            <div class="auth-title">AURA AI</div>
            <div class="auth-subtitle">
                Adaptive Uncertainty • Risk • Analytics
            </div>
        </div>
        """,
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

            # Log the new user in immediately.
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

    # Safety check: if the user was deleted/corrupted, return to login.
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
        # Hunter provides company/brand logos by domain.
        # Unlike Google's favicon service, this is intended to return
        # the company's actual brand logo rather than a generic favicon.
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

/* ==========================================================
   REFERENCE-STYLE TOP ACTION BAR
   - No chevrons/arrows on icon buttons
   - Bell and sun are icon-only, never boxed
   - Profile is a large circular initial with online dot
   ========================================================== */
.aura-header-actions {
    display:flex!important;
    align-items:center!important;
    justify-content:flex-end!important;
    width:100%!important;
    gap:6px!important;
}

/* ==========================================================
   IMPORTANT: remove Streamlit's native popover button chrome
   from the top-right controls. This is intentionally broad
   because Streamlit's DOM wrapper can change between versions.
   ========================================================== */
div[data-testid="column"]:has(.aura-top-marker) div[data-testid="stPopover"] button,
div[data-testid="column"]:has(.aura-top-marker) div[data-testid="stPopover"] > button,
div[data-testid="column"]:has(.aura-top-marker) div[data-testid="stPopover"] button[data-testid="stPopoverButton"] {
    box-shadow:none!important;
    outline:none!important;
    font-family:inherit!important;
}

/* Kill EVERY native chevron/icon that Streamlit adds to popovers. */
div[data-testid="column"]:has(.aura-top-marker) div[data-testid="stPopover"] button svg,
div[data-testid="column"]:has(.aura-top-marker) div[data-testid="stPopover"] button [data-testid="stIconMaterial"],
div[data-testid="column"]:has(.aura-top-marker) div[data-testid="stPopover"] button [data-testid="stIcon"],
div[data-testid="column"]:has(.aura-top-marker) div[data-testid="stPopover"] button span[data-testid="stIconMaterial"] {
    display:none!important;
    visibility:hidden!important;
    width:0!important;
    height:0!important;
    margin:0!important;
    padding:0!important;
}

/* ---------------- AI ENGINE ONLINE ---------------- */
div[data-testid="column"]:has(.aura-engine-marker) div[data-testid="stPopover"] button,
div[data-testid="column"]:has(.aura-engine-marker) div[data-testid="stPopover"] > button {
    width:142px!important;
    min-width:142px!important;
    height:42px!important;
    min-height:42px!important;
    padding:0 14px!important;
    border:1px solid rgba(34,211,238,.20)!important;
    border-radius:999px!important;
    background:rgba(7,15,24,.58)!important;
    color:#67e8f9!important;
    font-size:.68rem!important;
    font-weight:750!important;
    letter-spacing:.03em!important;
}
div[data-testid="column"]:has(.aura-engine-marker) div[data-testid="stPopover"] button:hover,
div[data-testid="column"]:has(.aura-engine-marker) div[data-testid="stPopover"] > button:hover {
    border-color:rgba(34,211,238,.45)!important;
    background:rgba(11,28,40,.85)!important;
}

/* ---------------- BELL: ICON ONLY ---------------- */
div[data-testid="column"]:has(.aura-bell-marker) div[data-testid="stPopover"] button,
div[data-testid="column"]:has(.aura-bell-marker) div[data-testid="stPopover"] > button,
div[data-testid="column"]:has(.aura-bell-marker) div[data-testid="stPopover"] button[data-testid="stPopoverButton"] {
    width:42px!important;
    min-width:42px!important;
    height:42px!important;
    min-height:42px!important;
    padding:0!important;
    margin:0!important;
    border:0!important;
    border-radius:0!important;
    background:transparent!important;
    color:#e8eef7!important;
    font-size:21px!important;
    font-weight:400!important;
}
div[data-testid="column"]:has(.aura-bell-marker) div[data-testid="stPopover"] button:hover,
div[data-testid="column"]:has(.aura-bell-marker) div[data-testid="stPopover"] > button:hover {
    border:0!important;
    background:transparent!important;
    color:#67e8f9!important;
    transform:scale(1.06);
}

/* ---------------- SUN: ICON ONLY ---------------- */
div[data-testid="column"]:has(.aura-sun-marker) div[data-testid="stPopover"] button,
div[data-testid="column"]:has(.aura-sun-marker) div[data-testid="stPopover"] > button,
div[data-testid="column"]:has(.aura-sun-marker) div[data-testid="stPopover"] button[data-testid="stPopoverButton"] {
    width:42px!important;
    min-width:42px!important;
    height:42px!important;
    min-height:42px!important;
    padding:0!important;
    margin:0!important;
    border:0!important;
    border-radius:0!important;
    background:transparent!important;
    color:#e8eef7!important;
    font-size:22px!important;
    font-weight:400!important;
}
div[data-testid="column"]:has(.aura-sun-marker) div[data-testid="stPopover"] button:hover,
div[data-testid="column"]:has(.aura-sun-marker) div[data-testid="stPopover"] > button:hover {
    border:0!important;
    background:transparent!important;
    color:#67e8f9!important;
    transform:scale(1.06);
}

/* ---------------- PROFILE: CIRCLE ONLY ---------------- */
div[data-testid="column"]:has(.aura-profile-marker) div[data-testid="stPopover"] button,
div[data-testid="column"]:has(.aura-profile-marker) div[data-testid="stPopover"] > button,
div[data-testid="column"]:has(.aura-profile-marker) div[data-testid="stPopover"] button[data-testid="stPopoverButton"] {
    position:relative!important;
    display:flex!important;
    align-items:center!important;
    justify-content:center!important;
    width:58px!important;
    min-width:58px!important;
    max-width:58px!important;
    height:58px!important;
    min-height:58px!important;
    max-height:58px!important;
    padding:0!important;
    margin:0!important;
    border-radius:50%!important;
    border:1px solid rgba(255,255,255,.18)!important;
    background:#263143!important;
    color:#f4f7fb!important;
    font-size:18px!important;
    font-weight:800!important;
    line-height:1!important;
    box-shadow:0 4px 16px rgba(0,0,0,.28)!important;
}
div[data-testid="column"]:has(.aura-profile-marker) div[data-testid="stPopover"] button:hover,
div[data-testid="column"]:has(.aura-profile-marker) div[data-testid="stPopover"] > button:hover {
    background:#303b50!important;
    border-color:rgba(34,211,238,.35)!important;
    box-shadow:0 0 22px rgba(34,211,238,.08)!important;
}

/* green online dot on the profile circle */
div[data-testid="column"]:has(.aura-profile-marker) div[data-testid="stPopover"] button::after,
div[data-testid="column"]:has(.aura-profile-marker) div[data-testid="stPopover"] > button::after {
    content:""!important;
    position:absolute!important;
    width:10px!important;
    height:10px!important;
    right:1px!important;
    bottom:1px!important;
    background:#22c55e!important;
    border:2px solid #07111c!important;
    border-radius:50%!important;
    display:block!important;
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
    div[data-testid="column"]:has(.aura-engine-marker) div[data-testid="stPopover"] button,
    div[data-testid="column"]:has(.aura-engine-marker) div[data-testid="stPopover"] > button {
        width:120px!important;
        min-width:120px!important;
        font-size:.61rem!important;
    }
    div[data-testid="column"]:has(.aura-profile-marker) div[data-testid="stPopover"] button,
    div[data-testid="column"]:has(.aura-profile-marker) div[data-testid="stPopover"] > button {
        width:52px!important;
        height:52px!important;
        min-width:52px!important;
        min-height:52px!important;
    }
}

/* ==========================================================
   FINAL TOP-HEADER OVERRIDE — TARGET THE ACTUAL STREAMLIT COLUMN
   The earlier .aura-bell/.aura-profile wrappers are siblings of
   Streamlit widgets, not their parents. These selectors use :has()
   to target the real column that contains each popover.
   ========================================================== */

/* Remove native popover caret/arrow ONLY from the four top controls. */
div[data-testid="column"]:has(.aura-top-marker) div[data-testid="stPopover"] button svg,
div[data-testid="column"]:has(.aura-top-marker) div[data-testid="stPopover"] button [data-testid="stIconMaterial"],
div[data-testid="column"]:has(.aura-top-marker) div[data-testid="stPopover"] button [data-testid="stIcon"],
div[data-testid="column"]:has(.aura-top-marker) div[data-testid="stPopover"] button span[aria-hidden="true"] {
    display:none!important;
    visibility:hidden!important;
    width:0!important;
    height:0!important;
    margin:0!important;
    padding:0!important;
}

/* Bell: absolutely no box, border, background, or arrow. */
div[data-testid="column"]:has(.aura-bell-marker) div[data-testid="stPopover"] button {
    width:42px!important; min-width:42px!important; max-width:42px!important;
    height:42px!important; min-height:42px!important; max-height:42px!important;
    padding:0!important; margin:0!important;
    border:0!important; outline:0!important;
    border-radius:0!important;
    background:transparent!important;
    box-shadow:none!important;
    color:#e8eef7!important;
    font-size:21px!important;
}
div[data-testid="column"]:has(.aura-bell-marker) div[data-testid="stPopover"] button:hover {
    border:0!important; background:transparent!important; box-shadow:none!important;
    color:#67e8f9!important; transform:scale(1.08);
}

/* Sun: icon only, no box and no arrow. */
div[data-testid="column"]:has(.aura-sun-marker) div[data-testid="stPopover"] button {
    width:42px!important; min-width:42px!important; max-width:42px!important;
    height:42px!important; min-height:42px!important; max-height:42px!important;
    padding:0!important; margin:0!important;
    border:0!important; outline:0!important;
    border-radius:0!important;
    background:transparent!important;
    box-shadow:none!important;
    color:#e8eef7!important;
    font-size:22px!important;
}
div[data-testid="column"]:has(.aura-sun-marker) div[data-testid="stPopover"] button:hover {
    border:0!important; background:transparent!important; box-shadow:none!important;
    color:#67e8f9!important; transform:scale(1.08);
}

/* Profile: H is the whole circular button. */
div[data-testid="column"]:has(.aura-profile-marker) div[data-testid="stPopover"] button {
    position:relative!important;
    display:flex!important; align-items:center!important; justify-content:center!important;
    width:58px!important; min-width:58px!important; max-width:58px!important;
    height:58px!important; min-height:58px!important; max-height:58px!important;
    padding:0!important; margin:0!important;
    border-radius:50%!important;
    border:1px solid rgba(255,255,255,.18)!important;
    background:#263143!important;
    color:#f4f7fb!important;
    font-size:18px!important; font-weight:800!important; line-height:1!important;
    box-shadow:0 4px 16px rgba(0,0,0,.28)!important;
}
div[data-testid="column"]:has(.aura-profile-marker) div[data-testid="stPopover"] button:hover {
    border-color:rgba(34,211,238,.35)!important;
    background:#303b50!important;
    box-shadow:0 0 22px rgba(34,211,238,.08)!important;
}
/* Green online dot. */
div[data-testid="column"]:has(.aura-profile-marker) div[data-testid="stPopover"] button::after {
    content:""!important; position:absolute!important;
    width:10px!important; height:10px!important; right:1px!important; bottom:1px!important;
    border-radius:50%!important; background:#22c55e!important;
    border:2px solid #07111c!important; display:block!important;
}

/* Keep the AI engine as a pill, but remove its native arrow. */
div[data-testid="column"]:has(.aura-engine-marker) div[data-testid="stPopover"] button {
    width:142px!important; min-width:142px!important; height:42px!important;
    padding:0 14px!important; border:1px solid rgba(34,211,238,.20)!important;
    border-radius:999px!important; background:rgba(7,15,24,.58)!important;
    color:#67e8f9!important; font-size:.68rem!important; font-weight:750!important;
}

/* Make sure marker spans never affect layout. */
.aura-top-marker { display:block!important; width:0!important; height:0!important; overflow:hidden!important; }
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
[data-testid="stPopoverBody"] { background:#0b111b!important; border:1px solid rgba(148,163,184,.12)!important; }
@media (max-width:900px) { .hero-grid{grid-template-columns:1fr;} .hero-visual{display:none;} .hero-final{padding:22px;} .hero-title-final{font-size:1.4rem;} .hero-description{max-width:none;} }
</style>
""", unsafe_allow_html=True)

if not require_authentication():
    st.stop()

logged_in_username = st.session_state.get("auth_username")
logged_in_user = get_user(logged_in_username) if logged_in_username else None
profile_name = (logged_in_user or {}).get("display_name") or logged_in_username or "User"
profile_initial = profile_name.strip()[0].upper() if profile_name.strip() else "U"

# ==========================================================
# HEADER — CLEAN REFERENCE-STYLE ACTION BAR
# ==========================================================
header_left, header_actions = st.columns([7.7, 2.3], gap="small")

with header_left:
    st.markdown(
        '<div class="aura-header-wrap">'
        '<div class="brand">AURA AI</div>'
        '<div class="brand-subtitle">ADAPTIVE UNCERTAINTY • RISK • ANALYTICS</div>'
        '</div>',
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
        st.markdown('<div class="aura-engine">', unsafe_allow_html=True)
        st.markdown('<span class="aura-top-marker aura-engine-marker"></span>', unsafe_allow_html=True)
        with st.popover("●  AI ENGINE ONLINE", use_container_width=True):
            st.markdown("### AI Engine")
            st.success("System online")
            st.caption(
                "AURA AI forecasting services are ready. "
                "The selected stock can be analysed using the trained LSTM "
                "and Monte Carlo uncertainty engine."
            )
            st.markdown("---")
            st.markdown(
                f'<div class="aura-setting-row">'
                f'<span class="aura-setting-label">Model</span>'
                f'<span class="aura-setting-value">LSTM + MC Dropout</span>'
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
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------------------
    # NOTIFICATIONS — icon only, clicking opens the panel
    # ------------------------------------------------------
    with bell_col:
        st.markdown('<div class="aura-bell">', unsafe_allow_html=True)
        st.markdown('<span class="aura-top-marker aura-bell-marker"></span>', unsafe_allow_html=True)
        with st.popover("🔔︎", use_container_width=True):
            # Use a bell character inside the panel; the trigger is
            # deliberately icon-only so Streamlit does not show a box.
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
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------------------
    # APPEARANCE — icon only, clicking opens a small panel
    # ------------------------------------------------------
    with sun_col:
        st.markdown('<div class="aura-sun">', unsafe_allow_html=True)
        st.markdown('<span class="aura-top-marker aura-sun-marker"></span>', unsafe_allow_html=True)
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
        st.markdown('<div class="aura-profile">', unsafe_allow_html=True)
        st.markdown('<span class="aura-top-marker aura-profile-marker"></span>', unsafe_allow_html=True)
        with st.popover(profile_initial, use_container_width=True):
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

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# SIDEBAR — MARKET CONTROL
# ==========================================================
st.sidebar.markdown('<div style="font-size:1.35rem;font-weight:850;letter-spacing:2px;margin-bottom:4px;">MARKET CONTROL</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="color:#68778c;font-size:.72rem;letter-spacing:1px;margin-bottom:20px;">SELECT AI MODEL</div>', unsafe_allow_html=True)
indian_stocks = [name for name, info in STOCKS.items() if info.get("market") == "India"]
us_stocks = [name for name, info in STOCKS.items() if info.get("market") == "USA"]
market = st.sidebar.radio("Market", ["🇮🇳 India", "🇺🇸 United States"], horizontal=False)
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
st.sidebar.markdown('<div style="margin-top:20px;color:#536176;font-size:.68rem;line-height:1.6;">Predictions are generated from the selected stock\'s dedicated trained model.</div>', unsafe_allow_html=True)

# ==========================================================
# DATA STATUS
# ==========================================================
stock_data = load_stock_data(selected_stock)
if stock_data is not None:
    valid_dates = stock_data["Date"].dropna() if "Date" in stock_data.columns else pd.Series(dtype="datetime64[ns]")
    last_date = valid_dates.iloc[-1].strftime("%d %b %Y") if len(valid_dates) else "Unavailable"
    data_rows = len(stock_data)
else:
    last_date = "Unavailable"; data_rows = 0
if st.session_state.get("live_stock") != selected_stock:
    st.session_state.pop("live_result", None)
    st.session_state.pop("prediction_time", None)

# ==========================================================
# HERO
# ==========================================================
hero_svg = """
<svg viewBox="0 0 330 155" preserveAspectRatio="none" aria-hidden="true">
<defs><linearGradient id="auraLine" x1="0" x2="1"><stop offset="0" stop-color="#22d3ee" stop-opacity=".15"/><stop offset=".48" stop-color="#67e8f9" stop-opacity=".95"/><stop offset="1" stop-color="#3b82f6" stop-opacity=".70"/></linearGradient><linearGradient id="auraFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#22d3ee" stop-opacity=".16"/><stop offset="1" stop-color="#22d3ee" stop-opacity="0"/></linearGradient></defs>
<g stroke="#334155" stroke-opacity=".18" stroke-width="1"><path d="M18 34H312M18 67H312M18 100H312M18 133H312"/><path d="M55 22V140M110 22V140M165 22V140M220 22V140M275 22V140"/></g>
<path d="M18 121 C48 112,55 118,78 101 S112 88,133 96 S164 74,183 82 S211 51,229 67 S256 43,276 55 S294 34,312 42 L312 140 L18 140 Z" fill="url(#auraFill)"/>
<path d="M18 121 C48 112,55 118,78 101 S112 88,133 96 S164 74,183 82 S211 51,229 67 S256 43,276 55 S294 34,312 42" fill="none" stroke="url(#auraLine)" stroke-width="2.8" stroke-linecap="round"/>
<path d="M183 82 L183 24 M229 67 L229 28" stroke="#67e8f9" stroke-opacity=".18" stroke-dasharray="3 4"/><circle cx="229" cy="67" r="4.5" fill="#67e8f9"/><circle cx="229" cy="67" r="10" fill="#67e8f9" fill-opacity=".08"/>
<text x="18" y="18" fill="#516276" font-size="7" font-family="Inter, sans-serif" letter-spacing="1.4">AI FORECAST ENGINE</text><text x="235" y="18" fill="#516276" font-size="7" font-family="Inter, sans-serif">MC • 95%</text>
</svg>"""
st.markdown(f'''<div class="hero-final"><div class="hero-grid"><div><div class="hero-label-final">AI MARKET INTELLIGENCE</div><div class="hero-main">{company_logo_html(selected_stock,ticker)}<div><div class="hero-title-final">{html.escape(str(company_name))}</div><div class="hero-ticker-final">{html.escape(str(ticker))} &nbsp; | &nbsp; {get_market_flag(config.get("market"))} {html.escape(str(config.get("market","")))} &nbsp; | &nbsp; Dedicated LSTM Model</div></div></div><div class="hero-description">Uncertainty-aware next-step forecasting using the selected stock's trained LSTM model with Monte Carlo Dropout risk estimation.</div><div class="hero-chip-row"><span class="hero-chip">LSTM FORECAST</span><span class="hero-chip">MC DROPOUT</span><span class="hero-chip">95% RANGE</span><span class="hero-chip">RISK AWARE</span></div></div><div class="hero-visual"><div class="hero-visual-label">AURA AI / MARKET TRAJECTORY</div>{hero_svg}</div></div></div>''', unsafe_allow_html=True)

# ==========================================================
# LIVE AI INTELLIGENCE
# ==========================================================
st.markdown('<div class="section-title">Live AI Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
run_prediction = st.button(f"⚡ RUN {str(selected_stock).upper()} AI ANALYSIS", use_container_width=True, key="run_final_prediction")
if run_prediction:
    with st.spinner(f"Running {selected_stock} LSTM + uncertainty engine..."):
        try:
            result = generate_live_prediction(selected_stock)
            if not isinstance(result, dict): raise ValueError("Prediction engine returned an invalid result format.")
            st.session_state["live_result"] = result
            st.session_state["live_stock"] = selected_stock
            st.session_state["prediction_time"] = time.strftime("%d %b %Y • %H:%M:%S")
            st.session_state["live_error"] = None
        except Exception as exc:
            st.session_state["live_result"] = None; st.session_state["live_stock"] = selected_stock; st.session_state["live_error"] = str(exc)
live_result = st.session_state.get("live_result") if st.session_state.get("live_stock") == selected_stock else None
live_error = st.session_state.pop("live_error", None)
if live_error: st.error(f"AI prediction failed: {live_error}")

# ==========================================================
# RESULTS
# ==========================================================
if live_result is not None:
    try:
        latest_price=float(live_result.get("latest_price",0)); predicted_price=float(live_result.get("predicted_price",0)); percentage_change=float(live_result.get("percentage_change",0)); uncertainty=abs(float(live_result.get("uncertainty",0))); lower_bound=float(live_result.get("lower_bound",predicted_price-uncertainty)); upper_bound=float(live_result.get("upper_bound",predicted_price+uncertainty)); signal=str(live_result.get("signal","Neutral")); risk=calculate_risk(uncertainty,predicted_price)
    except (TypeError,ValueError) as exc:
        st.error(f"Prediction result contains invalid numeric data: {exc}"); live_result=None

if live_result is not None:
    st.session_state["notifications"]=[
        {"type":"AI SIGNAL","message":f"{selected_stock}: {signal} signal generated.","time":st.session_state.get("prediction_time","Now")},
        {"type":"MARKET","message":f"Forecast: {format_currency(predicted_price,currency)} ({percentage_change:+.2f}%).","time":"Latest run"},
        {"type":"RISK","message":f"Model risk is {risk}; uncertainty ±{format_currency(uncertainty,currency)}.","time":"Latest run"},
    ]
    cols=st.columns(4)
    metric_data=[("Live Market Price",format_currency(latest_price,currency),"Latest market observation"),("AI Forecast",format_currency(predicted_price,currency),"Next predicted price"),("Expected Movement",("▲ " if percentage_change>=0 else "▼ ")+f"{percentage_change:+.2f}%","AI forecast vs current price"),("AI Uncertainty","±"+format_currency(uncertainty,currency),"Monte Carlo prediction spread")]
    for col,(label,value,small) in zip(cols,metric_data):
        with col: st.markdown(f'''<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-small">{small}</div></div>''',unsafe_allow_html=True)

    st.markdown('<div class="section-title">AI Decision Layer</div>',unsafe_allow_html=True); st.markdown('<div class="section-line"></div>',unsafe_allow_html=True)
    sk=signal.lower()
    signal_class,signal_text=("signal-positive","▲ POSITIVE") if "positive" in sk else (("signal-negative","▼ NEGATIVE") if "negative" in sk else ("signal-neutral","● NEUTRAL"))
    risk_symbol={"LOW":"◉","MODERATE":"◐","HIGH":"◑","VERY HIGH":"◉"}.get(risk,"●")
    d1,d2,d3=st.columns([1.1,1,1.5])
    with d1: st.markdown(f'''<div class="signal-card"><div class="metric-label">AI SIGNAL</div><div class="{signal_class}">{signal_text}</div><div class="metric-small">Based on model forecast</div></div>''',unsafe_allow_html=True)
    with d2: st.markdown(f'''<div class="signal-card"><div class="metric-label">MODEL RISK</div><div style="color:#f5f7fa;font-size:1.35rem;font-weight:800;margin-top:9px;">{risk_symbol} {risk}</div><div class="metric-small">{risk_message(risk)}</div></div>''',unsafe_allow_html=True)
    with d3: st.markdown(f'''<div class="signal-card"><div class="metric-label">95% PREDICTION RANGE</div><div style="color:#f5f7fa;font-size:1.18rem;font-weight:800;margin-top:9px;">{format_currency(lower_bound,currency)} &nbsp;|&nbsp; {format_currency(upper_bound,currency)}</div><div class="metric-small">Uncertainty interval produced by the AI engine</div></div>''',unsafe_allow_html=True)

    st.markdown('<div class="section-title">Market Trajectory</div>',unsafe_allow_html=True); st.markdown('<div class="section-line"></div>',unsafe_allow_html=True); st.markdown('<div class="chart-note">Historical price • AI forecast • shaded 95% prediction interval</div>',unsafe_allow_html=True)
    try:
        import plotly.graph_objects as go
        if stock_data is not None and "Close" in stock_data.columns and "Date" in stock_data.columns:
            chart_df=stock_data.dropna(subset=["Date","Close"]).tail(180).copy(); fig=go.Figure()
            fig.add_trace(go.Scatter(x=chart_df["Date"],y=chart_df["Close"],mode="lines",name="Historical Price",line=dict(width=2.5)))
            if len(chart_df):
                last_date_obj=chart_df["Date"].iloc[-1]; future_date=last_date_obj+pd.Timedelta(days=1)
                fig.add_trace(go.Scatter(x=[last_date_obj,future_date],y=[latest_price,predicted_price],mode="lines+markers",name="AI Forecast",line=dict(width=3,dash="dot"),marker=dict(size=9)))
                fig.add_trace(go.Scatter(x=[last_date_obj,future_date,future_date,last_date_obj],y=[latest_price,upper_bound,lower_bound,latest_price],fill="toself",fillcolor="rgba(34,211,238,0.10)",line=dict(width=0),hoverinfo="skip",name="95% Range"))
                fig.add_trace(go.Scatter(x=[future_date],y=[upper_bound],mode="markers",name="Upper Bound",marker=dict(size=7)))
                fig.add_trace(go.Scatter(x=[future_date],y=[lower_bound],mode="markers",name="Lower Bound",marker=dict(size=7)))
            fig.update_layout(height=470,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(color="#9aa8bb"),hovermode="x unified",xaxis=dict(showgrid=False,zeroline=False),yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.05)",zeroline=False,tickprefix=currency),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            st.plotly_chart(fig,use_container_width=True)
        else: st.warning("Historical chart data is unavailable.")
    except ImportError: st.warning("Plotly is required for the advanced chart.")
    except Exception as exc: st.warning(f"Unable to render the forecast chart: {exc}")

    st.markdown('<div class="section-title">Model Intelligence</div>',unsafe_allow_html=True); st.markdown('<div class="section-line"></div>',unsafe_allow_html=True)
    info=[("MODEL","LSTM","Long Short-Term Memory"),("UNCERTAINTY","MC DROPOUT","Monte Carlo simulations"),("DATA POINTS",f"{data_rows:,}","Historical observations"),("DATA THROUGH",last_date,"Latest stored observation")]
    info_cols=st.columns(4)
    for col,(label,value,small) in zip(info_cols,info):
        with col: st.markdown(f'''<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:1.05rem;">{value}</div><div class="metric-small">{small}</div></div>''',unsafe_allow_html=True)
    if st.session_state.get("prediction_time"): st.caption(f"Last AI analysis: {st.session_state['prediction_time']} • Selected model: {ticker}")
else:
    st.markdown('''<div class="ready-card"><div class="ready-orb">AI</div><div class="ready-title">AI Engine Ready</div><div class="ready-text">Select a stock from Market Control and run AI analysis to generate the next-price forecast, Monte Carlo uncertainty estimate, 95% prediction range and model risk assessment.</div></div>''',unsafe_allow_html=True)

# ==========================================================
# HISTORICAL MARKET DATA
# ==========================================================
st.markdown('<div class="section-title">Historical Market Data</div>',unsafe_allow_html=True); st.markdown('<div class="section-line"></div>',unsafe_allow_html=True)
if stock_data is not None:
    chart=create_historical_chart(stock_data)
    if chart is not None: st.plotly_chart(chart,use_container_width=True)
    with st.expander(f"View {selected_stock} raw market data"): st.dataframe(stock_data.tail(30),use_container_width=True,hide_index=True)
else: st.warning(f"No historical dataset found for {selected_stock}.")

st.markdown('''<div class="final-footer"><strong>AURA AI — INTELLIGENT STOCK ANALYTICS</strong><br>LSTM Forecasting • Monte Carlo Dropout • Uncertainty-Aware Prediction<br><br>Research &amp; Educational Use • Not Financial Advice</div>''',unsafe_allow_html=True)