import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)


# ============================================================
# 1. LOAD TEST DATA
# ============================================================

print("=" * 60)
print("LSTM MODEL EVALUATION")
print("=" * 60)

X_test = np.load(
    "data/processed/X_test.npy"
)

y_test = np.load(
    "data/processed/y_test.npy"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

model = load_model(
    "models/tcs_lstm_model.keras"
)

print("\nTrained LSTM model loaded successfully! ✅")


# ============================================================
# 3. LOAD SCALER
# ============================================================

scaler = joblib.load(
    "data/processed/scaler.pkl"
)

print("Scaler loaded successfully! ✅")


# ============================================================
# 4. MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions_scaled = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# 5. CONVERT BACK TO ORIGINAL PRICE
# ============================================================

actual_prices = scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()


predicted_prices = scaler.inverse_transform(
    predictions_scaled
).flatten()


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)

rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predicted_prices
    )
)

mape = mean_absolute_percentage_error(
    actual_prices,
    predicted_prices
) * 100

r2 = r2_score(
    actual_prices,
    predicted_prices
)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nMAE  : {mae:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"MAPE : {mape:.2f}%")

print(f"R²   : {r2:.4f}")


# ============================================================
# 8. SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


results_df = pd.DataFrame({
    "Actual_Price": actual_prices,
    "Predicted_Price": predicted_prices
})


results_df.to_csv(
    "results/predictions.csv",
    index=False
)


# ============================================================
# 9. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "MAPE",
        "R2"
    ],

    "Value": [
        mae,
        rmse,
        mape,
        r2
    ]
})


metrics_df.to_csv(
    "results/metrics.csv",
    index=False
)


# ============================================================
# 10. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    actual_prices,
    label="Actual Price"
)

plt.plot(
    predicted_prices,
    label="Predicted Price"
)

plt.title(
    "TCS Stock Price - Actual vs Predicted"
)

plt.xlabel(
    "Trading Days"
)

plt.ylabel(
    "Price (₹)"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "results/actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED SUCCESSFULLY! ✅")
print("=" * 60)

print("\nFiles created:")

print("results/predictions.csv")

print("results/metrics.csv")

print("results/actual_vs_predicted.png")

print("\nYour LSTM performance is ready for analysis! 📈")