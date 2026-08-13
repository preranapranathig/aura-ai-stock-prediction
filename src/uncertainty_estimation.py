import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "tcs_lstm_model.keras"
X_TEST_PATH = BASE_DIR / "data" / "processed" / "X_test.npy"
Y_TEST_PATH = BASE_DIR / "data" / "processed" / "y_test.npy"
SCALER_PATH = BASE_DIR / "data" / "processed" / "scaler.pkl"

RESULTS_DIR = BASE_DIR / "results"

OUTPUT_CSV = RESULTS_DIR / "uncertainty_predictions.csv"

# Keep this LOW for dashboard performance.
# 10 Monte Carlo passes are enough for this application.
N_ITERATIONS = 10


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

def load_resources():

    print("\n" + "=" * 60)
    print("AURA AI - UNCERTAINTY ENGINE")
    print("=" * 60)

    print(f"Model : {MODEL_PATH}")
    print(f"X test: {X_TEST_PATH}")
    print(f"Y test: {Y_TEST_PATH}")
    print(f"Scaler: {SCALER_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not X_TEST_PATH.exists():
        raise FileNotFoundError(f"X_test not found: {X_TEST_PATH}")

    if not Y_TEST_PATH.exists():
        raise FileNotFoundError(f"y_test not found: {Y_TEST_PATH}")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    print("\nLoading model...")

    model = load_model(MODEL_PATH)

    print("Model loaded successfully.")

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    scaler = joblib.load(SCALER_PATH)

    print(f"Test samples: {len(X_test)}")

    return model, X_test, y_test, scaler


# ============================================================
# MONTE CARLO DROPOUT
# ============================================================

def monte_carlo_prediction(model, X_test, iterations=10):

    predictions = []

    print("\nGenerating uncertainty predictions...")

    for i in range(iterations):

        # training=True keeps dropout active
        prediction = model(
            X_test,
            training=True
        ).numpy()

        prediction = np.asarray(prediction).reshape(-1)

        predictions.append(prediction)

        print(
            f"Monte Carlo pass {i + 1}/{iterations}"
        )

    predictions = np.asarray(predictions)

    return predictions


# ============================================================
# CALCULATE UNCERTAINTY
# ============================================================

def calculate_uncertainty(predictions):

    mean_prediction = np.mean(
        predictions,
        axis=0
    )

    std_prediction = np.std(
        predictions,
        axis=0
    )

    lower_bound = np.percentile(
        predictions,
        2.5,
        axis=0
    )

    upper_bound = np.percentile(
        predictions,
        97.5,
        axis=0
    )

    return (
        mean_prediction,
        std_prediction,
        lower_bound,
        upper_bound
    )


# ============================================================
# CONVERT BACK TO ORIGINAL PRICE SCALE
# ============================================================

def inverse_transform_predictions(
    scaler,
    values
):

    values = np.asarray(values).reshape(-1, 1)

    return scaler.inverse_transform(
        values
    ).flatten()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    actual_prices,
    mean_prediction,
    std_prediction,
    lower_bound,
    upper_bound
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = pd.DataFrame({
        "Actual_Price": actual_prices,
        "Predicted_Price": mean_prediction,
        "Uncertainty": std_prediction,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound
    })

    results.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\nResults saved:")
    print(OUTPUT_CSV)

    return results


# ============================================================
# MAIN
# ============================================================

def generate_uncertainty():

    try:

        model, X_test, y_test, scaler = load_resources()

        # ----------------------------------------------------
        # Monte Carlo predictions
        # ----------------------------------------------------

        predictions_scaled = monte_carlo_prediction(
            model,
            X_test,
            N_ITERATIONS
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        (
            mean_scaled,
            std_scaled,
            lower_scaled,
            upper_scaled
        ) = calculate_uncertainty(
            predictions_scaled
        )

        # ----------------------------------------------------
        # Convert predictions to original price
        # ----------------------------------------------------

        actual_prices = inverse_transform_predictions(
            scaler,
            y_test
        )

        mean_prediction = inverse_transform_predictions(
            scaler,
            mean_scaled
        )

        lower_bound = inverse_transform_predictions(
            scaler,
            lower_scaled
        )

        upper_bound = inverse_transform_predictions(
            scaler,
            upper_scaled
        )

        # Standard deviation is already in scaled space.
        # Convert it approximately using scaler scale.
        try:
            scale_value = float(
                np.asarray(scaler.scale_).flatten()[0]
            )

            std_prediction = std_scaled * scale_value

        except Exception:

            std_prediction = std_scaled

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        results = save_results(
            actual_prices,
            mean_prediction,
            std_prediction,
            lower_bound,
            upper_bound
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("UNCERTAINTY ESTIMATION COMPLETED")
        print("=" * 60)

        print(
            f"Latest actual price   : "
            f"{actual_prices[-1]:.2f}"
        )

        print(
            f"Latest prediction     : "
            f"{mean_prediction[-1]:.2f}"
        )

        print(
            f"95% lower bound      : "
            f"{lower_bound[-1]:.2f}"
        )

        print(
            f"95% upper bound      : "
            f"{upper_bound[-1]:.2f}"
        )

        print("=" * 60)

        return results

    except Exception as e:

        print("\nUNCERTAINTY ENGINE ERROR")
        print(str(e))

        raise


# ============================================================
# RUN ONLY WHEN FILE IS EXECUTED DIRECTLY
# ============================================================

if __name__ == "__main__":

    generate_uncertainty()