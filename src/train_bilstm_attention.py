# ============================================================
# AURA AI - BiLSTM + ATTENTION STOCK PRICE MODEL
# ALL 7 STOCKS | OHLCV | 60-DAY SEQUENCE
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    Attention,
    GlobalAveragePooling1D,
    Input,
    Layer
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
# SELF ATTENTION
# ============================================================

class SelfAttention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention()

    def call(self, inputs):
        return self.attention([inputs, inputs])


# ============================================================
# CONFIGURATION
# ============================================================

STOCKS = [
    "tcs",
    "infosys",
    "reliance",
    "hdfc_bank",
    "apple",
    "microsoft",
    "tesla"
]

SEQUENCE_LENGTH = 60
EPOCHS = 100
BATCH_SIZE = 32


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


print("=" * 75)
print("AURA AI - BiLSTM + ATTENTION MULTI-STOCK TRAINING")
print("=" * 75)

print("Architecture:")
print("Input → BiLSTM → Attention → Dense → Output")

print("\nFeatures:")
print("Open, High, Low, Close, Volume")

print(f"\nSequence Length : {SEQUENCE_LENGTH}")
print(f"Epochs          : {EPOCHS}")
print(f"Batch Size      : {BATCH_SIZE}")

print("\nStocks:")
print(", ".join(STOCKS))

print("=" * 75)


# ============================================================
# STORE ALL METRICS
# ============================================================

all_metrics = []


# ============================================================
# TRAIN EACH STOCK
# ============================================================

for stock in STOCKS:

    print("\n")
    print("=" * 75)
    print(f"TRAINING BiLSTM + ATTENTION : {stock.upper()}")
    print("=" * 75)

    processed_dir = os.path.join(
        "data",
        "processed",
        stock
    )

    X_train_path = os.path.join(
        processed_dir,
        "X_train.npy"
    )

    y_train_path = os.path.join(
        processed_dir,
        "y_train.npy"
    )

    X_test_path = os.path.join(
        processed_dir,
        "X_test.npy"
    )

    y_test_path = os.path.join(
        processed_dir,
        "y_test.npy"
    )

    scaler_path = os.path.join(
        processed_dir,
        "scaler.pkl"
    )


    # --------------------------------------------------------
    # CHECK REQUIRED FILES
    # --------------------------------------------------------

    required_files = [
        X_train_path,
        y_train_path,
        X_test_path,
        y_test_path,
        scaler_path
    ]

    missing = [
        path for path in required_files
        if not os.path.exists(path)
    ]

    if missing:

        print("\n[SKIPPED] Missing files:")

        for path in missing:
            print(" -", path)

        continue


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\n[1/6] Loading processed data...")

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    print("X_train :", X_train.shape)
    print("y_train :", y_train.shape)
    print("X_test  :", X_test.shape)
    print("y_test  :", y_test.shape)


    # --------------------------------------------------------
    # LOAD SCALER
    # --------------------------------------------------------

    print("\n[2/6] Loading scaler...")

    scaler = joblib.load(
        scaler_path
    )

    print("Scaler loaded.")


    # --------------------------------------------------------
    # BUILD BiLSTM + ATTENTION
    # --------------------------------------------------------

    print("\n[3/6] Building BiLSTM + Attention...")


    model = Sequential([

        Input(
            shape=(
                X_train.shape[1],
                X_train.shape[2]
            )
        ),

        # ----------------------------------------------------
        # BIDIRECTIONAL LSTM
        # ----------------------------------------------------

        Bidirectional(
            LSTM(
                64,
                return_sequences=True
            )
        ),

        Dropout(0.2),


        # ----------------------------------------------------
        # SECOND BiLSTM
        # ----------------------------------------------------

        Bidirectional(
            LSTM(
                32,
                return_sequences=True
            )
        ),

        Dropout(0.2),


        # ----------------------------------------------------
        # ATTENTION
        # ----------------------------------------------------

        SelfAttention(),


        # ----------------------------------------------------
        # GLOBAL REPRESENTATION
        # ----------------------------------------------------

        GlobalAveragePooling1D(),


        # ----------------------------------------------------
        # DENSE
        # ----------------------------------------------------

        Dense(
            16,
            activation="relu"
        ),

        Dropout(0.1),


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        Dense(1)

    ])


    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )


    print("\nModel created successfully.")


    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n[4/6] Training...")

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


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\n[5/6] Generating predictions...")

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    ).reshape(-1)


    # --------------------------------------------------------
    # INVERSE SCALING
    # --------------------------------------------------------

    actual_prices = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()


    predicted_prices = scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).flatten()


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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


    final_training_loss = (
        history.history["loss"][-1]
    )


    final_validation_loss = (
        history.history["val_loss"][-1]
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print(f"BiLSTM + ATTENTION RESULTS : {stock.upper()}")
    print("=" * 75)

    print(f"MAE              : {mae:.4f}")
    print(f"RMSE             : {rmse:.4f}")
    print(f"MAPE             : {mape:.4f}%")
    print(f"R²               : {r2:.4f}")
    print(f"Epochs Completed : {epochs_completed}")

    print("=" * 75)


    # --------------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------------

    model_path = os.path.join(
        "models",
        f"{stock}_bilstm_attention_model.keras"
    )


    csv_path = os.path.join(
        "results",
        f"{stock}_bilstm_attention_metrics.csv"
    )


    json_path = os.path.join(
        "results",
        f"{stock}_bilstm_attention_metrics.json"
    )


    predictions_path = os.path.join(
        "results",
        f"{stock}_bilstm_attention_predictions.npz"
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model.save(
        model_path
    )


    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    np.savez(
        predictions_path,
        actual=actual_prices,
        predicted=predicted_prices
    )


    # --------------------------------------------------------
    # METRICS RECORD
    # --------------------------------------------------------

    metric_record = {

        "stock": stock,

        "status": "trained",

        "model": model_path,

        "architecture": (
            "BiLSTM + Attention"
        ),

        "features": (
            "Open, High, Low, Close, Volume"
        ),

        "sequence_length": (
            SEQUENCE_LENGTH
        ),

        "epochs_completed": (
            epochs_completed
        ),

        "MAE": float(mae),

        "RMSE": float(rmse),

        "MAPE": float(mape),

        "R2": float(r2),

        "final_training_loss": (
            float(final_training_loss)
        ),

        "final_validation_loss": (
            float(final_validation_loss)
        )
    }


    all_metrics.append(
        metric_record
    )


    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    pd.DataFrame(
        [metric_record]
    ).to_csv(
        csv_path,
        index=False
    )


    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metric_record,
            file,
            indent=4
        )


    print("\nSaved:")

    print(model_path)
    print(csv_path)
    print(json_path)
    print(predictions_path)


# ============================================================
# MASTER METRICS
# ============================================================

print("\n")
print("=" * 75)
print("CREATING MASTER BiLSTM + ATTENTION METRICS")
print("=" * 75)


if all_metrics:

    master_df = pd.DataFrame(
        all_metrics
    )


    master_csv = (
        "results/"
        "all_stock_bilstm_attention_metrics.csv"
    )


    master_json = (
        "results/"
        "all_stock_bilstm_attention_metrics.json"
    )


    master_df.to_csv(
        master_csv,
        index=False
    )


    with open(
        master_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_metrics,
            file,
            indent=4
        )


    print("\nMASTER RESULTS:")

    print(
        master_df[
            [
                "stock",
                "MAE",
                "RMSE",
                "MAPE",
                "R2"
            ]
        ].to_string(
            index=False
        )
    )


    print("\nSaved:")
    print(master_csv)
    print(master_json)


else:

    print(
        "\nNo stocks were successfully trained."
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 75)
print("BiLSTM + ATTENTION MULTI-STOCK TRAINING FINISHED")
print("=" * 75)

print(
    f"Successful stocks : {len(all_metrics)} / {len(STOCKS)}"
)

print("=" * 75)