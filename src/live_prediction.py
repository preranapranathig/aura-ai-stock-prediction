"""
AURA AI - Live LSTM + Monte Carlo Dropout Prediction Engine

IMPORTANT:
- This module changes ONLY the AI backend.
- The Streamlit dashboard/UI is not modified.
- It uses the trained LSTM + Attention model produced by the
  AURA AI training pipeline.

PERFORMANCE FIX (this version):
- _mc_predict() previously ran 100 SEPARATE sequential forward passes
  in a Python for-loop. Each call carries its own Python + TensorFlow
  graph-execution overhead, which on CPU (especially with an Attention
  layer) can easily add up to a minute or more — a very likely cause
  of "prediction timed out" errors.
- Now it runs all 100 Monte Carlo samples as ONE batched forward pass
  (repeat the input 100 times along the batch axis, call the model
  once). This is dramatically faster and produces identical results,
  since each row in the batch still gets its own independent dropout
  mask.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Attention, Layer


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
STOCK_DIR = BASE_DIR / "data" / "stocks"
RESULTS_DIR = BASE_DIR / "results"

SEQUENCE_LENGTH = 60
MC_SAMPLES = 30
CONFIDENCE_Z = 1.96

FEATURES = ["Open", "High", "Low", "Close", "Volume"]

# The training pipeline currently defines these stock folders/files.
STOCK_FILE_MAP = {
    "TCS": "tcs.csv",
    "Infosys": "infosys.csv",
    "Reliance": "reliance.csv",
    "HDFC Bank": "hdfc_bank.csv",
    "Apple": "apple.csv",
    "Microsoft": "microsoft.csv",
    "Tesla": "tesla.csv",
}


# ============================================================
# CUSTOM LAYER
# Must match the training architecture exactly.
# ============================================================

class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention()

    def call(self, inputs):
        return self.attention([inputs, inputs])


# ============================================================
# HELPERS
# ============================================================

def _safe_name(stock_name: str) -> str:
    return str(stock_name).strip().lower().replace(" ", "_")


def _stock_key(stock_name: str) -> str:
    value = str(stock_name).strip()

    aliases = {
        "TCS": "TCS",
        "Infosys": "Infosys",
        "INFY": "Infosys",
        "Reliance": "Reliance",
        "RELIANCE": "Reliance",
        "HDFC Bank": "HDFC Bank",
        "HDFCBANK": "HDFC Bank",
        "Apple": "Apple",
        "AAPL": "Apple",
        "Microsoft": "Microsoft",
        "MSFT": "Microsoft",
        "Tesla": "Tesla",
        "TSLA": "Tesla",
    }

    return aliases.get(value, value)


def _candidate_model_paths(stock_name: str):
    key = _stock_key(stock_name)
    safe = _safe_name(key)

    return [
        MODEL_DIR / f"{safe}_lstm_attention_model.keras",
        MODEL_DIR / f"{safe}_lstm_model.keras",
        MODEL_DIR / f"{safe}_model.keras",
        MODEL_DIR / f"{safe}_lstm_attention_model.h5",
        MODEL_DIR / f"{safe}_lstm_model.h5",
        MODEL_DIR / f"{safe}_model.h5",
    ]


def _candidate_scaler_paths(stock_name: str):
    key = _stock_key(stock_name)
    safe = _safe_name(key)

    # IMPORTANT:
    # The live model uses 5 OHLCV features:
    # Open, High, Low, Close, Volume.
    #
    # Some stock folders also contain a one-feature scaler.pkl.
    # That scaler is NOT compatible with the live OHLCV sequence.
    # Prefer feature_scaler.pkl, which is the 5-feature scaler.
    return [
        PROCESSED_DIR / safe / "feature_scaler.pkl",
        PROCESSED_DIR / safe / "scaler.pkl",
    ]


def _candidate_data_paths(stock_name: str):
    key = _stock_key(stock_name)
    safe = _safe_name(key)

    candidates = []

    mapped = STOCK_FILE_MAP.get(key)
    if mapped:
        candidates.append(STOCK_DIR / mapped)

    candidates.extend([
        STOCK_DIR / f"{safe}.csv",
        STOCK_DIR / f"{safe}_stock_data.csv",
        BASE_DIR / "data" / f"{safe}.csv",
        BASE_DIR / "data" / f"{safe}_stock_data.csv",
    ])

    # Preserve order while removing duplicates.
    unique = []
    seen = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)

    return unique


def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _find_model(stock_name: str):
    path = _first_existing(_candidate_model_paths(stock_name))
    if path is None:
        key = _stock_key(stock_name)
        expected = _candidate_model_paths(stock_name)[0]
        raise FileNotFoundError(
            f"No trained model was found for {key}. "
            f"Expected a model such as: {expected}"
        )
    return path


def _find_scaler(stock_name: str):
    key = _stock_key(stock_name)

    # Only accept a scaler trained for the 5 OHLCV input features.
    # This prevents accidentally loading a one-feature scaler.pkl.
    for path in _candidate_scaler_paths(stock_name):
        if not path.exists():
            continue

        try:
            scaler = joblib.load(path)
            n_features = getattr(scaler, "n_features_in_", None)

            if n_features == len(FEATURES):
                return path

        except Exception:
            continue

    expected = _candidate_scaler_paths(stock_name)[0]

    raise FileNotFoundError(
        f"No compatible 5-feature scaler was found for {key}. "
        f"Expected a scaler such as: {expected}"
    )


def _find_stock_data(stock_name: str):
    path = _first_existing(_candidate_data_paths(stock_name))
    if path is None:
        key = _stock_key(stock_name)
        raise FileNotFoundError(
            f"No historical OHLCV CSV was found for {key}. "
            f"Expected a file under {STOCK_DIR}."
        )
    return path


# ============================================================
# DATA PREPARATION
# ============================================================

def _load_latest_sequence(stock_name: str):
    data_path = _find_stock_data(stock_name)

    df = pd.read_csv(data_path)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Date"] + FEATURES
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{data_path.name} is missing required columns: {missing}. "
            f"Required columns are: {required}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=["Date"] + FEATURES)
          .sort_values("Date")
          .drop_duplicates(subset=["Date"], keep="last")
          .reset_index(drop=True)
    )

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Only {len(df)} valid OHLCV rows are available for {stock_name}; "
            f"{SEQUENCE_LENGTH} rows are required."
        )

    latest = df.iloc[-1]

    return df, latest, data_path


def _inverse_close(scaler, values):
    """
    Convert scaled Close values back to the original Close-price scale.

    Training scaler feature order:
        0 = Open
        1 = High
        2 = Low
        3 = Close
        4 = Volume
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    dummy = np.zeros((len(values), 5), dtype=np.float32)
    dummy[:, 3] = values

    restored = scaler.inverse_transform(dummy)
    return restored[:, 3]


# ============================================================
# MODEL LOADING
# ============================================================

_MODEL_CACHE = {}
_SCALER_CACHE = {}


def _load_model(stock_name: str):
    key = _stock_key(stock_name)

    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model_path = _find_model(stock_name)

    # compile=False avoids requiring the original training optimizer/loss
    # configuration. The architecture itself is preserved in the .keras file.
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"SelfAttention": SelfAttention},
        compile=False,
    )

    _MODEL_CACHE[key] = model
    return model


def _load_scaler(stock_name: str):
    """
    Load ONLY the 5-feature OHLCV scaler.

    The live model requires:
        Open, High, Low, Close, Volume

    Therefore a scaler with n_features_in_ == 1
    must never be used here.
    """
    key = _stock_key(stock_name)

    if key in _SCALER_CACHE:
        return _SCALER_CACHE[key]

    scaler_path = PROCESSED_DIR / _safe_name(key) / "feature_scaler.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"5-feature scaler not found for {key}.\n"
            f"Expected: {scaler_path}"
        )

    scaler = joblib.load(scaler_path)

    n_features = getattr(scaler, "n_features_in_", None)

    if n_features != len(FEATURES):
        raise ValueError(
            f"Invalid scaler for {key}: "
            f"expected {len(FEATURES)} features "
            f"({FEATURES}), but scaler expects {n_features}."
        )

    _SCALER_CACHE[key] = scaler

    return scaler

# ============================================================
# MONTE CARLO DROPOUT
# ============================================================

def _mc_predict(model, sequence, samples=MC_SAMPLES):
    """
    Run Monte Carlo Dropout as ONE batched forward pass instead of
    `samples` sequential calls.

    `sequence` has shape (1, seq_len, n_features). We repeat it along
    the batch axis to (samples, seq_len, n_features) and call the model
    ONCE with training=True. Because Dropout applies an independent
    random mask per batch row, each of the `samples` rows in the output
    is still a genuinely independent stochastic sample — statistically
    identical to the old loop, just far faster (one TensorFlow call
    instead of `samples` of them).
    """
    x = tf.convert_to_tensor(sequence, dtype=tf.float32)  # (1, seq_len, features)
    batch_x = tf.repeat(x, repeats=int(samples), axis=0)  # (samples, seq_len, features)

    output = model(batch_x, training=True)
    values = np.asarray(output.numpy()).reshape(-1)

    if len(values) == 0:
        raise RuntimeError("The model returned an empty prediction.")

    return values.astype(np.float32)


# ============================================================
# SIGNAL / RISK
# ============================================================

def _signal_from_change(change_pct):
    if change_pct >= 1.0:
        return "Positive"
    if change_pct <= -1.0:
        return "Negative"
    return "Neutral"


def _risk_from_uncertainty(uncertainty, predicted_price):
    if predicted_price <= 0:
        return "UNKNOWN"

    uncertainty_pct = abs(uncertainty / predicted_price) * 100.0

    if uncertainty_pct < 1.0:
        return "LOW"
    if uncertainty_pct < 2.5:
        return "MODERATE"
    if uncertainty_pct < 5.0:
        return "HIGH"
    return "VERY HIGH"


# ============================================================
# PUBLIC API USED BY THE EXISTING DASHBOARD
# ============================================================

def generate_live_prediction(stock_name):
    """
    Generate one-day-ahead prediction for the selected stock.

    Return keys are intentionally compatible with the existing
    Streamlit dashboard:
        latest_price
        predicted_price
        percentage_change
        uncertainty
        lower_bound
        upper_bound
        signal
        risk
        samples
        model
        sequence_length
        data_through
        data_file
    """

    key = _stock_key(stock_name)

    # Load the exact trained components used by the model.
    model = _load_model(key)
    scaler = _load_scaler(key)

    df, latest_row, data_path = _load_latest_sequence(key)

    # Use the same OHLCV feature order as training.
    raw_sequence = (
        df[FEATURES]
        .tail(SEQUENCE_LENGTH)
        .to_numpy(dtype=np.float32)
    )

    scaled_sequence = scaler.transform(raw_sequence)

    # Shape: (1, 60, 5)
    model_input = np.expand_dims(
        scaled_sequence.astype(np.float32),
        axis=0,
    )

    # Monte Carlo stochastic predictions in scaled space (now one
    # batched call instead of MC_SAMPLES sequential calls — see the
    # docstring on _mc_predict for why this is much faster).
    mc_scaled = _mc_predict(
        model,
        model_input,
        samples=MC_SAMPLES,
    )

    # Convert all stochastic predictions to rupee/dollar price space.
    mc_prices = _inverse_close(
        scaler,
        mc_scaled,
    )

    mc_prices = mc_prices[np.isfinite(mc_prices)]

    if len(mc_prices) < 2:
        raise RuntimeError(
            "Monte Carlo prediction did not produce enough valid samples."
        )

    latest_price = float(latest_row["Close"])
    predicted_price = float(np.mean(mc_prices))

    # Sample standard deviation is the uncertainty of the MC prediction
    # distribution. The dashboard already interprets this as ± uncertainty.
    uncertainty = float(np.std(mc_prices, ddof=1))

    lower_bound = float(
        predicted_price - CONFIDENCE_Z * uncertainty
    )
    upper_bound = float(
        predicted_price + CONFIDENCE_Z * uncertainty
    )

    percentage_change = float(
        ((predicted_price - latest_price) / latest_price) * 100.0
    ) if latest_price else 0.0

    signal = _signal_from_change(percentage_change)
    risk = _risk_from_uncertainty(
        uncertainty,
        predicted_price,
    )

    # Prevent an impossible negative lower price.
    lower_bound = max(0.0, lower_bound)

    return {
        "stock": key,
        "status": "success",
        "model": "LSTM + Attention",
        "uncertainty_method": "Monte Carlo Dropout",
        "horizon": "1 Day",
        "sequence_length": SEQUENCE_LENGTH,
        "latest_price": latest_price,
        "predicted_price": predicted_price,
        "percentage_change": percentage_change,
        "uncertainty": uncertainty,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "signal": signal,
        "risk": risk,
        "samples": mc_prices.astype(float).tolist(),
        "mc_samples": int(len(mc_prices)),
        "data_through": pd.Timestamp(latest_row["Date"]).strftime("%d %b %Y"),
        "data_file": str(data_path),
        "rows_used": int(len(df)),
    }


# ============================================================
# OPTIONAL QUICK TEST
# Run:
#     python src/live_prediction.py
# ============================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    import time as _time

    try:
        _start = _time.time()
        result = generate_live_prediction("TCS")
        _elapsed = _time.time() - _start

        print("=" * 70)
        print("AURA AI - LIVE BACKEND TEST")
        print("=" * 70)
        print(f"Stock             : {result['stock']}")
        print(f"Model             : {result['model']}")
        print(f"Uncertainty       : {result['uncertainty_method']}")
        print(f"Data Through      : {result['data_through']}")
        print(f"Latest Price      : {result['latest_price']:.2f}")
        print(f"Predicted Price   : {result['predicted_price']:.2f}")
        print(f"Movement          : {result['percentage_change']:+.2f}%")
        print(f"Uncertainty       : ±{result['uncertainty']:.2f}")
        print(f"95% Lower Bound   : {result['lower_bound']:.2f}")
        print(f"95% Upper Bound   : {result['upper_bound']:.2f}")
        print(f"Signal            : {result['signal']}")
        print(f"Risk              : {result['risk']}")
        print(f"MC Samples        : {result['mc_samples']}")
        print(f"Elapsed seconds   : {_elapsed:.2f}")
        print("=" * 70)

    except Exception as exc:
        print("AURA AI backend test failed:")
        print(exc)
        raise