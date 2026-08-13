import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from tensorflow.keras.models import load_model

from stock_config import STOCKS


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 60
MONTE_CARLO_RUNS = 100


# ============================================================
# GET STOCK CONFIGURATION
# ============================================================

def get_stock_config(stock_name):

    if stock_name not in STOCKS:
        raise ValueError(
            f"Unknown stock: {stock_name}"
        )

    config = STOCKS[stock_name]

    return {
        "name": config["name"],
        "ticker": config["ticker"],
        "market": config["market"],
        "currency": config["currency"],
        "model": config["model"],
        "scaler": config["scaler"]
    }


# ============================================================
# DOWNLOAD LATEST MARKET DATA
# ============================================================

def download_latest_data(ticker):

    print("=" * 60)
    print(f"Fetching latest data for {ticker}")
    print("=" * 60)

    data = yf.download(
        ticker,
        period="6mo",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            f"No market data received for {ticker}"
        )

    # Handle Yahoo Finance MultiIndex
    if isinstance(data.columns, pd.MultiIndex):

        data.columns = data.columns.get_level_values(0)

    data = data.dropna()

    if "Close" not in data.columns:

        raise ValueError(
            "Close price column not found."
        )

    return data


# ============================================================
# MONTE CARLO DROPOUT PREDICTION
# ============================================================

def monte_carlo_prediction(model, input_data):

    # Create 100 copies of the same input
    batch_input = np.repeat(
        input_data,
        MONTE_CARLO_RUNS,
        axis=0
    )

    # One batched stochastic prediction
    predictions = model(
        batch_input,
        training=True
    ).numpy()

    # Convert to 1D array
    predictions = predictions.reshape(-1)

    return predictions


# ============================================================
# GENERATE LIVE PREDICTION
# ============================================================

def generate_live_prediction(stock_name="TCS"):

    print("\n")
    print("=" * 70)
    print("       AI LIVE STOCK PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. GET CONFIGURATION
    # --------------------------------------------------------

    config = get_stock_config(
        stock_name
    )

    ticker = config["ticker"]
    model_path = config["model"]
    scaler_path = config["scaler"]

    print(f"\nStock       : {config['name']}")
    print(f"Ticker      : {ticker}")
    print(f"Market      : {config['market']}")
    print(f"Model       : {model_path}")
    print(f"Scaler      : {scaler_path}")

    # --------------------------------------------------------
    # 2. CHECK MODEL
    # --------------------------------------------------------

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    # --------------------------------------------------------
    # 3. CHECK SCALER
    # --------------------------------------------------------

    if not os.path.exists(scaler_path):

        raise FileNotFoundError(
            f"Scaler not found:\n{scaler_path}"
        )

    # --------------------------------------------------------
    # 4. DOWNLOAD LATEST DATA
    # --------------------------------------------------------

    data = download_latest_data(
        ticker
    )

    close_prices = data[
        "Close"
    ].values.reshape(-1, 1)

    if len(close_prices) < SEQUENCE_LENGTH:

        raise ValueError(
            f"Not enough data for prediction. "
            f"Required: {SEQUENCE_LENGTH}, "
            f"Available: {len(close_prices)}"
        )

    latest_price = float(
        close_prices[-1][0]
    )

    # --------------------------------------------------------
    # 5. LOAD SCALER
    # --------------------------------------------------------

    scaler = joblib.load(
        scaler_path
    )

    # --------------------------------------------------------
    # 6. SCALE DATA
    # --------------------------------------------------------

    scaled_prices = scaler.transform(
        close_prices
    )

    # --------------------------------------------------------
    # 7. GET LAST 60 DAYS
    # --------------------------------------------------------

    sequence = scaled_prices[
        -SEQUENCE_LENGTH:
    ]

    X = sequence.reshape(
        1,
        SEQUENCE_LENGTH,
        1
    )

    # --------------------------------------------------------
    # 8. LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading LSTM model...")

    model = load_model(
        model_path
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # 9. MONTE CARLO DROPOUT
    # --------------------------------------------------------
    print(
        f"\nRunning "
        f"{MONTE_CARLO_RUNS} "
        f"Monte Carlo simulations..."
    )

    print("Generating stochastic predictions...")

    mc_predictions = monte_carlo_prediction(
        model,
        X
    )

    print("Monte Carlo simulations completed.")


    # --------------------------------------------------------
    # 10. CONVERT BACK TO REAL PRICE
    # --------------------------------------------------------

    mc_predictions = mc_predictions.reshape(
        -1,
        1
    )

    real_predictions = scaler.inverse_transform(
        mc_predictions
    ).flatten()

    # --------------------------------------------------------
    # 11. CALCULATE STATISTICS
    # --------------------------------------------------------

    predicted_price = float(
        np.mean(real_predictions)
    )

    uncertainty = float(
        np.std(real_predictions)
    )

    # 95% interval
    lower_bound = float(
        np.percentile(
            real_predictions,
            2.5
        )
    )

    upper_bound = float(
        np.percentile(
            real_predictions,
            97.5
        )
    )

    # --------------------------------------------------------
    # 12. CALCULATE PERCENTAGE CHANGE
    # --------------------------------------------------------

    percentage_change = (
        (predicted_price - latest_price)
        / latest_price
    ) * 100

    # --------------------------------------------------------
    # 13. GENERATE SIGNAL
    # --------------------------------------------------------

    if percentage_change > 1:

        signal = "Positive"

    elif percentage_change < -1:

        signal = "Negative"

    else:

        signal = "Neutral"

    # --------------------------------------------------------
    # 14. DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LIVE PREDICTION RESULT")
    print("=" * 70)

    print(
        f"\nCurrent Price    : "
        f"{config['currency']}{latest_price:,.2f}"
    )

    print(
        f"Predicted Price  : "
        f"{config['currency']}{predicted_price:,.2f}"
    )

    print(
        f"Expected Change  : "
        f"{percentage_change:+.2f}%"
    )

    print(
        f"Uncertainty      : "
        f"±{config['currency']}{uncertainty:,.2f}"
    )

    print(
        f"95% Lower Bound  : "
        f"{config['currency']}{lower_bound:,.2f}"
    )

    print(
        f"95% Upper Bound  : "
        f"{config['currency']}{upper_bound:,.2f}"
    )

    print(
        f"AI Signal        : "
        f"{signal}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 15. RETURN RESULT
    # --------------------------------------------------------

    return {
        "stock_name": stock_name,
        "company_name": config["name"],
        "ticker": ticker,
        "market": config["market"],
        "currency": config["currency"],

        "latest_price": latest_price,

        "predicted_price": predicted_price,

        "percentage_change": percentage_change,

        "uncertainty": uncertainty,

        "lower_bound": lower_bound,

        "upper_bound": upper_bound,

        "signal": signal,

        "monte_carlo_runs": MONTE_CARLO_RUNS
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = generate_live_prediction(
        "TCS"
    )

    print("\nPrediction completed successfully.")