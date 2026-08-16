import numpy as np
import pandas as pd
import joblib
import yfinance as yf

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# CONFIGURATION
# ============================================================

TICKER = "TCS.NS"

SEQUENCE_LENGTH = 60

MONTE_CARLO_RUNS = 100

MODEL_PATH = "models/tcs_lstm_model.keras"

SCALER_PATH = "data/processed/scaler.pkl"


# ============================================================
# START
# ============================================================

print("=" * 60)
print("LIVE TCS STOCK PRICE PREDICTION")
print("=" * 60)


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("\nLoading trained LSTM model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully! ✅")

print(
    f"Model input shape: {model.input_shape}"
)


# ============================================================
# 2. DOWNLOAD LATEST DATA
# ============================================================

print(f"\nDownloading latest data for {TICKER}...")

data = yf.download(
    TICKER,
    period="6mo",
    interval="1d",
    auto_adjust=False,
    progress=False
)


if data.empty:
    raise RuntimeError(
        "Could not download stock data."
    )


# Handle yfinance MultiIndex

if isinstance(data.columns, pd.MultiIndex):

    data.columns = data.columns.get_level_values(0)


data = data.dropna()


print(
    f"Downloaded {len(data)} trading-day records."
)


# ============================================================
# 3. CHECK DATA
# ============================================================

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]


missing = [
    col for col in FEATURES
    if col not in data.columns
]


if missing:

    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


if len(data) < SEQUENCE_LENGTH:

    raise RuntimeError(
        f"Need at least {SEQUENCE_LENGTH} "
        f"trading days."
    )


# ============================================================
# 4. EXTRACT 5 FEATURES
# ============================================================

feature_data = data[
    FEATURES
].astype(float)


latest_actual_price = float(
    feature_data["Close"].iloc[-1]
)


# ============================================================
# 5. LOAD TARGET SCALER
# ============================================================

print("\nLoading scaler...")

target_scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded successfully! ✅")


# ============================================================
# 6. SCALE FIVE INPUT FEATURES
# ============================================================

print("\nPreparing 5-feature input...")

feature_scaler = MinMaxScaler()

scaled_features = feature_scaler.fit_transform(
    feature_data.values
)


# ============================================================
# 7. LAST 60 DAYS
# ============================================================

latest_sequence = scaled_features[
    -SEQUENCE_LENGTH:
]


# ============================================================
# 8. CREATE LSTM INPUT
# ============================================================

X_input = latest_sequence.reshape(
    1,
    SEQUENCE_LENGTH,
    5
)


print(
    f"Input shape: {X_input.shape}"
)


# ============================================================
# SAFETY CHECK
# ============================================================

expected_features = model.input_shape[-1]


if X_input.shape[-1] != expected_features:

    raise RuntimeError(
        f"Feature mismatch!\n"
        f"Model expects: {expected_features}\n"
        f"Input contains: {X_input.shape[-1]}"
    )


# ============================================================
# 9. NORMAL PREDICTION
# ============================================================

print("\nGenerating prediction...")

normal_prediction_scaled = model.predict(
    X_input,
    verbose=0
)


normal_prediction = target_scaler.inverse_transform(
    normal_prediction_scaled.reshape(-1, 1)
)[0][0]


# ============================================================
# 10. MONTE CARLO DROPOUT
# ============================================================

print(
    f"\nRunning {MONTE_CARLO_RUNS} "
    "Monte Carlo simulations..."
)


mc_predictions = []


for _ in range(MONTE_CARLO_RUNS):

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
# 11. STATISTICS
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
# 12. CONVERT TO PRICE
# ============================================================

mean_prediction = target_scaler.inverse_transform(
    np.array([[mean_scaled]])
)[0][0]


lower_bound = target_scaler.inverse_transform(
    np.array([[lower_scaled]])
)[0][0]


upper_bound = target_scaler.inverse_transform(
    np.array([[upper_scaled]])
)[0][0]


# ============================================================
# 13. UNCERTAINTY
# ============================================================

price_range = (
    target_scaler.data_max_[0]
    - target_scaler.data_min_[0]
)


uncertainty = (
    std_scaled * price_range
)


# ============================================================
# 14. EXPECTED MOVEMENT
# ============================================================

percentage_change = (
    (mean_prediction - latest_actual_price)
    / latest_actual_price
) * 100


# ============================================================
# 15. SIGNAL
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