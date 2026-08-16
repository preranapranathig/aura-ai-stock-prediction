# ============================================================
# AURA AI - BiLSTM STOCK PRICE MODEL
# TCS | OHLCV | 60-DAY SEQUENCE
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

STOCK = "TCS"

SEQUENCE_LENGTH = 60
EPOCHS = 100
BATCH_SIZE = 32

X_TRAIN_PATH = "data/processed/tcs/X_train.npy"
Y_TRAIN_PATH = "data/processed/tcs/y_train.npy"

X_TEST_PATH = "data/processed/tcs/X_test.npy"
Y_TEST_PATH = "data/processed/tcs/y_test.npy"

SCALER_PATH = "data/processed/tcs/scaler.pkl"

MODEL_PATH = "models/tcs_bilstm_model.keras"

CSV_PATH = "results/tcs_bilstm_metrics.csv"
JSON_PATH = "results/tcs_bilstm_metrics.json"
PREDICTIONS_PATH = "results/tcs_bilstm_predictions.npz"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AURA AI - BiLSTM STOCK PRICE MODEL")
print("=" * 70)

print(f"Stock             : {STOCK}")
print(f"Sequence Length   : {SEQUENCE_LENGTH}")
print(f"Features           : Open, High, Low, Close, Volume")
print(f"Epochs             : {EPOCHS}")
print(f"Batch Size         : {BATCH_SIZE}")
print("=" * 70)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

print("\n[1/6] Loading processed data...")

X_train = np.load(X_TRAIN_PATH)
y_train = np.load(Y_TRAIN_PATH)

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_test shape  : {y_test.shape}")


# ============================================================
# 2. LOAD SCALER
# ============================================================

print("\n[2/6] Loading scaler...")

scaler = joblib.load(SCALER_PATH)

print("Scaler loaded successfully.")


# ============================================================
# 3. BUILD BiLSTM MODEL
# ============================================================

print("\n[3/6] Building BiLSTM model...")

model = Sequential([
    Bidirectional(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    ),

    Dropout(0.2),

    Bidirectional(
        LSTM(
            32,
            return_sequences=False
        )
    ),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])


model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


model.summary()


# ============================================================
# 4. TRAIN MODEL
# ============================================================

print("\n[4/6] Training BiLSTM...")
print("=" * 70)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)


history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ============================================================
# 5. PREDICTION + EVALUATION
# ============================================================

print("\n[5/6] Evaluating BiLSTM...")

predictions_scaled = model.predict(
    X_test,
    verbose=0
).reshape(-1)


# ------------------------------------------------------------
# IMPORTANT:
# Scaler contains 5 features:
# Open, High, Low, Close, Volume
#
# y_test contains only scaled Close values.
# Therefore we inverse-transform using the CLOSE column.
# ------------------------------------------------------------

def inverse_close(values):
    values = np.asarray(values).reshape(-1)

    dummy = np.zeros(
        (len(values), 5)
    )

    # Close is column index 3
    dummy[:, 3] = values

    restored = scaler.inverse_transform(dummy)

    return restored[:, 3]


actual_prices = inverse_close(y_test)

predicted_prices = inverse_close(predictions_scaled)


# ============================================================
# METRICS
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


epochs_completed = len(
    history.history["loss"]
)

final_training_loss = history.history["loss"][-1]

final_validation_loss = history.history["val_loss"][-1]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("BiLSTM MODEL RESULTS - TCS")
print("=" * 70)

print(f"MAE             : {mae:.4f}")
print(f"RMSE            : {rmse:.4f}")
print(f"MAPE            : {mape:.4f}%")
print(f"R²              : {r2:.4f}")
print(f"Epochs Completed : {epochs_completed}")

print("=" * 70)


# ============================================================
# 6. SAVE EVERYTHING
# ============================================================

print("\n[6/6] Saving model and results...")

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

model.save(
    MODEL_PATH
)

print(f"Model saved      : {MODEL_PATH}")


# ------------------------------------------------------------
# SAVE PREDICTIONS
# ------------------------------------------------------------

np.savez(
    PREDICTIONS_PATH,
    actual=actual_prices,
    predicted=predicted_prices
)

print(f"Predictions saved: {PREDICTIONS_PATH}")


# ------------------------------------------------------------
# SAVE CSV METRICS
# ------------------------------------------------------------

metrics_df = pd.DataFrame([
    {
        "stock": STOCK,
        "status": "trained",
        "model": MODEL_PATH,
        "epochs_completed": epochs_completed,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "final_training_loss": final_training_loss,
        "final_validation_loss": final_validation_loss
    }
])


metrics_df.to_csv(
    CSV_PATH,
    index=False
)

print(f"Metrics CSV saved: {CSV_PATH}")


# ------------------------------------------------------------
# SAVE JSON METRICS
# ------------------------------------------------------------

metrics_json = {
    "stock": STOCK,
    "status": "trained",
    "model": MODEL_PATH,
    "architecture": "BiLSTM",
    "features": [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ],
    "sequence_length": SEQUENCE_LENGTH,
    "epochs_completed": epochs_completed,
    "MAE": float(mae),
    "RMSE": float(rmse),
    "MAPE": float(mape),
    "R2": float(r2),
    "final_training_loss": float(final_training_loss),
    "final_validation_loss": float(final_validation_loss)
}


with open(
    JSON_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics_json,
        file,
        indent=4
    )


print(f"Metrics JSON saved: {JSON_PATH}")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("BiLSTM TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nFiles created:")

print(f"1. {MODEL_PATH}")
print(f"2. {CSV_PATH}")
print(f"3. {JSON_PATH}")
print(f"4. {PREDICTIONS_PATH}")

print("\nTCS BiLSTM is ready for comparison with LSTM.")
print("=" * 70)