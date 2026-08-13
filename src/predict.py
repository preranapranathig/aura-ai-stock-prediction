import numpy as np
import pandas as pd
import joblib
import yfinance as yf

from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

TICKER = "TCS.NS"

SEQUENCE_LENGTH = 60

MONTE_CARLO_RUNS = 100

MODEL_PATH = "models/tcs_lstm_model.keras"

SCALER_PATH = "data/processed/scaler.pkl"


# ============================================================
# 1. DOWNLOAD LATEST STOCK DATA
# ============================================================

print("=" * 60)
print("LIVE TCS STOCK PRICE PREDICTION")
print("=" * 60)

print(f"\nDownloading latest data for {TICKER}...")

data = yf.download(
    TICKER,
    period="6mo",
    interval="1d",
    auto_adjust=False,
    progress=False
)


# ============================================================
# 2. CHECK DATA
# ============================================================

if data.empty:

    raise RuntimeError(
        "Could not download stock data."
    )


# Handle yfinance multi-level columns

if isinstance(data.columns, pd.MultiIndex):

    data.columns = data.columns.get_level_values(0)


data = data.dropna()


print(
    f"Downloaded {len(data)} trading-day records."
)


# ============================================================
# 3. CHECK FOR ENOUGH DATA
# ============================================================

if len(data) < SEQUENCE_LENGTH:

    raise RuntimeError(
        f"Not enough data. "
        f"Need at least {SEQUENCE_LENGTH} trading days."
    )


# ============================================================
# 4. GET CLOSE PRICES
# ============================================================

close_prices = data["Close"].values.reshape(-1, 1)


latest_actual_price = float(
    close_prices[-1][0]
)


# ============================================================
# 5. LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_PATH
)


# ============================================================
# 6. SCALE LATEST DATA
# ============================================================

scaled_prices = scaler.transform(
    close_prices
)


# ============================================================
# 7. TAKE LAST 60 DAYS
# ============================================================

latest_sequence = scaled_prices[
    -SEQUENCE_LENGTH:
]


# ============================================================
# 8. RESHAPE FOR LSTM
# ============================================================

X_input = latest_sequence.reshape(
    1,
    SEQUENCE_LENGTH,
    1
)


print(
    f"\nInput shape: {X_input.shape}"
)


# ============================================================
# 9. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained LSTM model...")

model = load_model(
    MODEL_PATH
)


print("Model loaded successfully! ✅")


# ============================================================
# 10. NORMAL PREDICTION
# ============================================================

normal_prediction_scaled = model.predict(
    X_input,
    verbose=0
)


normal_prediction = scaler.inverse_transform(
    normal_prediction_scaled
)[0][0]


# ============================================================
# 11. MONTE CARLO DROPOUT
# ============================================================

print(
    f"\nRunning {MONTE_CARLO_RUNS} "
    "Monte Carlo simulations..."
)


mc_predictions = []


for i in range(MONTE_CARLO_RUNS):

    # training=True keeps Dropout active

    prediction = model(
        X_input,
        training=True
    ).numpy()

    mc_predictions.append(
        prediction[0][0]
    )


mc_predictions = np.array(
    mc_predictions
)


# ============================================================
# 12. CALCULATE PREDICTION STATISTICS
# ============================================================

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


# ============================================================
# 13. CONVERT BACK TO PRICE
# ============================================================

mean_prediction = scaler.inverse_transform(
    np.array([[mean_scaled]])
)[0][0]


lower_bound = scaler.inverse_transform(
    np.array([[lower_scaled]])
)[0][0]


upper_bound = scaler.inverse_transform(
    np.array([[upper_scaled]])
)[0][0]


# Convert uncertainty to original price scale

price_range = (
    scaler.data_max_[0]
    - scaler.data_min_[0]
)

uncertainty = (
    std_scaled * price_range
)


# ============================================================
# 14. CALCULATE EXPECTED MOVEMENT
# ============================================================

percentage_change = (
    (mean_prediction - latest_actual_price)
    / latest_actual_price
) * 100


# ============================================================
# 15. GENERATE SIMPLE MODEL SIGNAL
# ============================================================

if percentage_change > 1:

    signal = "📈 Positive"

elif percentage_change < -1:

    signal = "📉 Negative"

else:

    signal = "➡️ Neutral"


# ============================================================
# 16. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LIVE PREDICTION RESULTS")
print("=" * 60)

print(
    f"\nLatest actual price:"
    f" ₹{latest_actual_price:,.2f}"
)

print(
    f"\nPredicted price:"
    f" ₹{mean_prediction:,.2f}"
)

print(
    f"\nExpected movement:"
    f" {percentage_change:+.2f}%"
)

print(
    f"\nModel signal:"
    f" {signal}"
)

print(
    f"\nPrediction uncertainty:"
    f" ±₹{uncertainty:,.2f}"
)

print(
    "\n95% Prediction Interval:"
)

print(
    f"₹{lower_bound:,.2f}"
    f" → "
    f"₹{upper_bound:,.2f}"
)


print("\n" + "=" * 60)
print("LIVE PREDICTION COMPLETED! ✅")
print("=" * 60)