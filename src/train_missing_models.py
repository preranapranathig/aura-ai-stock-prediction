# ============================================================
# AURA AI - MISSING MODEL TRAINING ENGINE
# BiLSTM + LSTM Attention
# ALL 7 STOCKS | OHLCV | 60-DAY SEQUENCE
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D,
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
# STOCKS
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


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 100
BATCH_SIZE = 32
SEQUENCE_LENGTH = 60


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# HELPER: TRAIN ONE MODEL
# ============================================================

def train_model(
    stock,
    architecture,
    X_train,
    y_train,
    X_test,
    y_test,
    scaler
):

    print("\n")
    print("=" * 80)
    print(
        f"{stock.upper()} | {architecture}"
    )
    print("=" * 80)


    # ========================================================
    # BUILD MODEL
    # ========================================================

    if architecture == "BiLSTM":

        model = Sequential([

            Input(
                shape=(
                    X_train.shape[1],
                    X_train.shape[2]
                )
            ),

            Bidirectional(
                LSTM(
                    64,
                    return_sequences=True
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

            Dense(
                16,
                activation="relu"
            ),

            Dense(1)
        ])


    elif architecture == "LSTM + Attention":

        model = Sequential([

            Input(
                shape=(
                    X_train.shape[1],
                    X_train.shape[2]
                )
            ),

            LSTM(
                64,
                return_sequences=True
            ),

            Dropout(0.2),

            LSTM(
                32,
                return_sequences=True
            ),

            Dropout(0.2),

            SelfAttention(),

            GlobalAveragePooling1D(),

            Dense(
                16,
                activation="relu"
            ),

            Dropout(0.1),

            Dense(1)
        ])


    else:

        raise ValueError(
            f"Unknown architecture: {architecture}"
        )


    # ========================================================
    # COMPILE
    # ========================================================

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )


    # ========================================================
    # CALLBACKS
    # ========================================================

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


    # ========================================================
    # TRAIN
    # ========================================================

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


    # ========================================================
    # PREDICTION
    # ========================================================

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    ).reshape(-1)


    # ========================================================
    # INVERSE TRANSFORMATION
    # ========================================================

    actual_prices = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()


    predicted_prices = scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).flatten()


    # ========================================================
    # METRICS
    # ========================================================

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


    # ========================================================
    # FILE NAME PREFIX
    # ========================================================

    architecture_key = (
        "bilstm"
        if architecture == "BiLSTM"
        else "lstm_attention"
    )


    model_path = os.path.join(
        "models",
        f"{stock}_{architecture_key}_model.keras"
    )


    csv_path = os.path.join(
        "results",
        f"{stock}_{architecture_key}_metrics.csv"
    )


    json_path = os.path.join(
        "results",
        f"{stock}_{architecture_key}_metrics.json"
    )


    predictions_path = os.path.join(
        "results",
        f"{stock}_{architecture_key}_predictions.npz"
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model.save(
        model_path
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    np.savez(
        predictions_path,
        actual=actual_prices,
        predicted=predicted_prices
    )


    # ========================================================
    # METRIC RECORD
    # ========================================================

    record = {

        "stock": stock,

        "status": "trained",

        "model": model_path,

        "architecture": architecture,

        "features": [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ],

        "sequence_length": SEQUENCE_LENGTH,

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


    # ========================================================
    # SAVE CSV
    # ========================================================

    pd.DataFrame(
        [record]
    ).to_csv(
        csv_path,
        index=False
    )


    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            record,
            file,
            indent=4
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    print("\nRESULTS")

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"MAPE : {mape:.4f}%"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    print(
        f"Epochs: {epochs_completed}"
    )

    print("\nSaved:")

    print(
        model_path
    )

    print(
        csv_path
    )

    print(
        json_path
    )

    print(
        predictions_path
    )


    return record


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("AURA AI - MISSING MODEL TRAINING")
print("=" * 80)

print(
    "\nModels:"
)

print(
    "1. BiLSTM"
)

print(
    "2. LSTM + Attention"
)

print(
    "\nStocks:"
)

print(
    ", ".join(STOCKS)
)

print(
    "\nFeatures:"
)

print(
    "Open, High, Low, Close, Volume"
)

print(
    "\nSequence Length:",
    SEQUENCE_LENGTH
)

print("=" * 80)


all_bilstm = []
all_attention = []


# ============================================================
# PROCESS EACH STOCK
# ============================================================

for stock in STOCKS:

    print("\n")
    print("#" * 80)
    print(
        f"PROCESSING {stock.upper()}"
    )
    print("#" * 80)


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


    # ========================================================
    # CHECK FILES
    # ========================================================

    required_files = [

        X_train_path,

        y_train_path,

        X_test_path,

        y_test_path,

        scaler_path
    ]


    missing = [

        path

        for path in required_files

        if not os.path.exists(path)
    ]


    if missing:

        print(
            "\n[SKIPPED] Missing files:"
        )

        for path in missing:

            print(
                " -",
                path
            )

        continue


    # ========================================================
    # LOAD DATA
    # ========================================================

    X_train = np.load(
        X_train_path
    )

    y_train = np.load(
        y_train_path
    )

    X_test = np.load(
        X_test_path
    )

    y_test = np.load(
        y_test_path
    )


    scaler = joblib.load(
        scaler_path
    )


    print(
        "\nData shapes:"
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test :",
        X_test.shape
    )


    # ========================================================
    # TRAIN BiLSTM
    # ========================================================

    bilstm_record = train_model(

        stock,

        "BiLSTM",

        X_train,

        y_train,

        X_test,

        y_test,

        scaler
    )


    all_bilstm.append(
        bilstm_record
    )


    # ========================================================
    # TRAIN LSTM + ATTENTION
    # ========================================================

    attention_record = train_model(

        stock,

        "LSTM + Attention",

        X_train,

        y_train,

        X_test,

        y_test,

        scaler
    )


    all_attention.append(
        attention_record
    )


# ============================================================
# MASTER BiLSTM RESULTS
# ============================================================

if all_bilstm:

    bilstm_df = pd.DataFrame(
        all_bilstm
    )


    bilstm_df.to_csv(
        "results/all_stock_bilstm_metrics.csv",
        index=False
    )


    with open(
        "results/all_stock_bilstm_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_bilstm,
            file,
            indent=4
        )


# ============================================================
# MASTER LSTM + ATTENTION RESULTS
# ============================================================

if all_attention:

    attention_df = pd.DataFrame(
        all_attention
    )


    attention_df.to_csv(
        "results/all_stock_lstm_attention_metrics.csv",
        index=False
    )


    with open(
        "results/all_stock_lstm_attention_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_attention,
            file,
            indent=4
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("MISSING MODEL TRAINING COMPLETED")
print("=" * 80)

print(
    f"\nBiLSTM trained:"
    f" {len(all_bilstm)} / {len(STOCKS)}"
)

print(
    f"LSTM + Attention trained:"
    f" {len(all_attention)} / {len(STOCKS)}"
)


print("\nMASTER FILES:")

print(
    "results/all_stock_bilstm_metrics.csv"
)

print(
    "results/all_stock_bilstm_metrics.json"
)

print(
    "results/all_stock_lstm_attention_metrics.csv"
)

print(
    "results/all_stock_lstm_attention_metrics.json"
)


print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)