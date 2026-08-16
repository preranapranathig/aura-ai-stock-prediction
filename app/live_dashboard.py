import streamlit as st
import numpy as np
import pandas as pd
import joblib
import yfinance as yf

from tensorflow.keras.models import load_model


# ============================================================
# CONFIG
# ============================================================

TICKER = "TCS.NS"

SEQUENCE_LENGTH = 60

MODEL_PATH = "models/tcs_lstm_model.keras"
SCALER_PATH = "data/processed/scaler.pkl"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="AURA AI - Stock Prediction",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 AURA AI - TCS Stock Prediction")

st.caption(
    "LSTM-based stock prediction with Monte Carlo uncertainty estimation"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=300)
def get_stock_data():

    data = yf.download(
        TICKER,
        period="6mo",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    return load_model(MODEL_PATH)


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def get_scaler():

    return joblib.load(SCALER_PATH)


# ============================================================
# PREDICTION
# ============================================================

def make_prediction(data, model, scaler):

    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    feature_data = data[features].astype(float)

    latest_price = float(
        feature_data["Close"].iloc[-1]
    )

    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    from sklearn.preprocessing import MinMaxScaler

    feature_scaler = MinMaxScaler()

    scaled_data = feature_scaler.fit_transform(
        feature_data.values
    )

    # --------------------------------------------------------
    # Last 60 days
    # --------------------------------------------------------

    sequence = scaled_data[-SEQUENCE_LENGTH:]

    X = sequence.reshape(
        1,
        SEQUENCE_LENGTH,
        5
    )

    # --------------------------------------------------------
    # Normal prediction
    # --------------------------------------------------------

    prediction_scaled = model.predict(
        X,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    )[0][0]

    # --------------------------------------------------------
    # Monte Carlo
    # --------------------------------------------------------

    mc_predictions = []

    for _ in range(100):

        pred = model(
            X,
            training=True
        ).numpy()

        mc_predictions.append(
            pred[0][0]
        )

    mc_predictions = np.array(
        mc_predictions
    )

    mean_scaled = np.mean(
        mc_predictions
    )

    std_scaled = np.std(
        mc_predictions
    )

    lower_scaled = np.percentile(
        mc_predictions,
        2.5
    )

    upper_scaled = np.percentile(
        mc_predictions,
        97.5
    )

    # --------------------------------------------------------
    # Convert back to price
    # --------------------------------------------------------

    mean_prediction = scaler.inverse_transform(
        np.array([[mean_scaled]])
    )[0][0]

    lower_bound = scaler.inverse_transform(
        np.array([[lower_scaled]])
    )[0][0]

    upper_bound = scaler.inverse_transform(
        np.array([[upper_scaled]])
    )[0][0]

    # --------------------------------------------------------
    # Uncertainty
    # --------------------------------------------------------

    price_range = (
        scaler.data_max_[0]
        - scaler.data_min_[0]
    )

    uncertainty = (
        std_scaled * price_range
    )

    # --------------------------------------------------------
    # Movement
    # --------------------------------------------------------

    percentage_change = (
        (mean_prediction - latest_price)
        / latest_price
    ) * 100

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    if percentage_change > 1:

        signal = "📈 Positive"

    elif percentage_change < -1:

        signal = "📉 Negative"

    else:

        signal = "➡️ Neutral"

    return {
        "latest": latest_price,
        "prediction": mean_prediction,
        "change": percentage_change,
        "signal": signal,
        "uncertainty": uncertainty,
        "lower": lower_bound,
        "upper": upper_bound
    }


# ============================================================
# MAIN
# ============================================================

try:

    with st.spinner("Loading live TCS market data..."):

        data = get_stock_data()

    model = get_model()

    scaler = get_scaler()

    result = make_prediction(
        data,
        model,
        scaler
    )


    # ========================================================
    # TOP METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Current TCS Price",
            f"₹{result['latest']:,.2f}"
        )


    with col2:

        st.metric(
            "AI Predicted Price",
            f"₹{result['prediction']:,.2f}"
        )


    with col3:

        st.metric(
            "Expected Movement",
            f"{result['change']:+.2f}%"
        )


    with col4:

        st.metric(
            "AI Signal",
            result["signal"]
        )


    st.divider()


    # ========================================================
    # UNCERTAINTY
    # ========================================================

    st.subheader("🤖 AI Prediction & Uncertainty")


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            f"""
            **Predicted Price**

            ₹{result['prediction']:,.2f}

            **Uncertainty**

            ±₹{result['uncertainty']:,.2f}
            """
        )


    with col2:

        st.success(
            f"""
            **95% Prediction Interval**

            ₹{result['lower']:,.2f}
            
            ↓

            ₹{result['upper']:,.2f}
            """
        )


    # ========================================================
    # PRICE CHART
    # ========================================================

    st.subheader("📊 TCS Historical Price")


    chart_data = data[
        ["Close"]
    ].tail(90)

    st.line_chart(
        chart_data
    )


    # ========================================================
    # PREDICTION COMPARISON
    # ========================================================

    st.subheader("🎯 Prediction Summary")


    summary = pd.DataFrame({

        "Metric": [
            "Current Price",
            "Predicted Price",
            "Expected Change",
            "Uncertainty",
            "Lower Bound",
            "Upper Bound"
        ],

        "Value": [

            f"₹{result['latest']:,.2f}",

            f"₹{result['prediction']:,.2f}",

            f"{result['change']:+.2f}%",

            f"±₹{result['uncertainty']:,.2f}",

            f"₹{result['lower']:,.2f}",

            f"₹{result['upper']:,.2f}"
        ]
    })


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # MODEL INFO
    # ========================================================

    st.divider()

    st.subheader("🧠 Model Information")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.write("**Model:** LSTM")


    with col2:

        st.write("**Input:** 60 trading days")


    with col3:

        st.write("**Features:** OHLC + Volume")


    st.caption(
        "⚠️ This is an AI prediction system for educational/research purposes and is not financial advice."
    )


except Exception as e:

    st.error(
        f"Dashboard error: {e}"
    )

    st.exception(e)