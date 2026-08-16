import os
import json
import csv
import numpy as np
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# AURA AI
# MULTI-STOCK LSTM TRAINING ENGINE
# ============================================================

print("=" * 80)
print("          AURA AI - MULTI-STOCK LSTM TRAINING ENGINE")
print("=" * 80)


# ============================================================
# STOCK CONFIGURATION
# ============================================================

STOCKS = {
    "TCS": "tcs",
    "Infosys": "infosys",
    "Reliance": "reliance",
    "HDFC Bank": "hdfc_bank",
    "Apple": "apple",
    "Microsoft": "microsoft",
    "Tesla": "tesla"
}


# ============================================================
# TRAINING SETTINGS
# ============================================================

EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.10
PATIENCE = 12

# IMPORTANT:
# We are now training models using the newly preprocessed data.
# Therefore TCS must also be retrained if its preprocessing
# structure has changed from the old model.

TRAIN_TCS = True


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# LSTM MODEL
# ============================================================

def build_model(input_shape):

    model = Sequential(
        [
            Input(shape=input_shape),

            LSTM(
                64,
                return_sequences=True
            ),

            Dropout(0.20),

            LSTM(
                32,
                return_sequences=False
            ),

            Dropout(0.20),

            Dense(
                16,
                activation="relu"
            ),

            Dense(1)
        ]
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# ============================================================
# INVERSE TRANSFORM TARGET
# ============================================================

def inverse_transform_target(
    scaler,
    values,
    n_features
):
    """
    Convert scaled Close prices back to original price.

    Supports both:
    1. One-feature scaler
    2. Five-feature OHLCV scaler

    Expected OHLCV order:
    Open, High, Low, Close, Volume
    """

    values = np.asarray(values).reshape(-1)

    # --------------------------------------------------------
    # Case 1: scaler contains only one feature
    # --------------------------------------------------------

    if n_features == 1:

        restored = scaler.inverse_transform(
            values.reshape(-1, 1)
        )

        return restored.reshape(-1)

    # --------------------------------------------------------
    # Case 2: scaler contains multiple features
    # --------------------------------------------------------

    temp = np.zeros(
        (len(values), n_features)
    )

    # Close is feature index 3 in OHLCV
    CLOSE_INDEX = 3

    temp[:, CLOSE_INDEX] = values

    restored = scaler.inverse_transform(temp)

    return restored[:, CLOSE_INDEX]


# ============================================================
# SAVE METRICS TO CSV
# ============================================================

def save_metrics_csv(
    stock_name,
    folder_name,
    metrics
):

    metrics_path = os.path.join(
        "results",
        f"{folder_name}_metrics.csv"
    )

    file_exists = os.path.exists(
        metrics_path
    )

    fieldnames = [
        "stock",
        "status",
        "model",
        "epochs_completed",
        "MAE",
        "RMSE",
        "MAPE",
        "R2",
        "final_training_loss",
        "final_validation_loss"
    ]

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerow(metrics)

    print(
        f"Metrics CSV saved: {metrics_path}"
    )

    return metrics_path


# ============================================================
# TRAIN ONE STOCK
# ============================================================

def train_stock(
    stock_name,
    folder_name
):

    print("\n")
    print("=" * 80)
    print(f"TRAINING MODEL: {stock_name}")
    print("=" * 80)

    # --------------------------------------------------------
    # MODEL PATH
    # --------------------------------------------------------

    model_path = os.path.join(
        "models",
        f"{folder_name}_lstm_model.keras"
    )

    # --------------------------------------------------------
    # DATA PATHS
    # --------------------------------------------------------

    processed_folder = os.path.join(
        "data",
        "processed",
        folder_name
    )

    X_train_path = os.path.join(
        processed_folder,
        "X_train.npy"
    )

    X_test_path = os.path.join(
        processed_folder,
        "X_test.npy"
    )

    y_train_path = os.path.join(
        processed_folder,
        "y_train.npy"
    )

    y_test_path = os.path.join(
        processed_folder,
        "y_test.npy"
    )

    scaler_path = os.path.join(
        processed_folder,
        "scaler.pkl"
    )

    # --------------------------------------------------------
    # CHECK REQUIRED FILES
    # --------------------------------------------------------

    required_files = [
        X_train_path,
        X_test_path,
        y_train_path,
        y_test_path,
        scaler_path
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):

            print("\nERROR: Missing file:")
            print(file_path)

            return {
                "stock": stock_name,
                "status": "failed",
                "model": model_path
            }

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading processed data...")

    X_train = np.load(
        X_train_path
    )

    X_test = np.load(
        X_test_path
    )

    y_train = np.load(
        y_train_path
    )

    y_test = np.load(
        y_test_path
    )

    scaler = joblib.load(
        scaler_path
    )

    # --------------------------------------------------------
    # DISPLAY DATA SHAPES
    # --------------------------------------------------------

    print(
        f"X_train shape : {X_train.shape}"
    )

    print(
        f"y_train shape : {y_train.shape}"
    )

    print(
        f"X_test shape  : {X_test.shape}"
    )

    print(
        f"y_test shape  : {y_test.shape}"
    )

    # --------------------------------------------------------
    # DETERMINE FEATURE COUNT
    # --------------------------------------------------------

    if len(X_train.shape) != 3:

        raise ValueError(
            "X_train must have shape "
            "(samples, timesteps, features)."
        )

    n_features = X_train.shape[2]

    print(
        f"Number of input features: {n_features}"
    )

    if n_features == 5:

        print(
            "Feature configuration: "
            "OHLCV"
        )

    elif n_features == 1:

        print(
            "Feature configuration: "
            "Close price only"
        )

    else:

        print(
            "WARNING: Unexpected number "
            "of features."
        )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print("\nBuilding LSTM architecture...")

    model = build_model(
        (
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    print("\nModel architecture:")

    model.summary()

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n")
    print("-" * 80)
    print(
        f"Starting training for {stock_name}"
    )
    print(
        f"Maximum epochs : {EPOCHS}"
    )
    print(
        f"Batch size     : {BATCH_SIZE}"
    )
    print("-" * 80)

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[
            early_stopping,
            reduce_lr
        ],
        verbose=1
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    print("\nSaving model...")

    model.save(
        model_path
    )

    print(
        f"Model saved: {model_path}"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print("\nEvaluating model...")

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    )

    predictions_scaled = (
        predictions_scaled
        .reshape(-1)
    )

    actual_scaled = (
        y_test
        .reshape(-1)
    )

    # --------------------------------------------------------
    # CONVERT TO REAL PRICE
    # --------------------------------------------------------

    predictions = inverse_transform_target(
        scaler,
        predictions_scaled,
        n_features
    )

    actual = inverse_transform_target(
        scaler,
        actual_scaled,
        n_features
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    # --------------------------------------------------------
    # MAPE
    # --------------------------------------------------------

    non_zero_mask = actual != 0

    if np.any(non_zero_mask):

        mape = (
            np.mean(
                np.abs(
                    (
                        actual[non_zero_mask]
                        -
                        predictions[non_zero_mask]
                    )
                    /
                    actual[non_zero_mask]
                )
            )
            * 100
        )

    else:

        mape = 0.0

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    r2 = r2_score(
        actual,
        predictions
    )

    # --------------------------------------------------------
    # TRAINING INFORMATION
    # --------------------------------------------------------

    epochs_completed = len(
        history.history["loss"]
    )

    final_train_loss = float(
        history.history["loss"][-1]
    )

    final_val_loss = float(
        history.history["val_loss"][-1]
    )

    # --------------------------------------------------------
    # METRICS DICTIONARY
    # --------------------------------------------------------

    metrics = {

        "stock": stock_name,

        "status": "trained",

        "model": model_path,

        "epochs_completed":
            epochs_completed,

        "MAE":
            float(mae),

        "RMSE":
            float(rmse),

        "MAPE":
            float(mape),

        "R2":
            float(r2),

        "final_training_loss":
            final_train_loss,

        "final_validation_loss":
            final_val_loss
    }

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    json_path = os.path.join(
        "results",
        f"{folder_name}_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    save_metrics_csv(
        stock_name,
        folder_name,
        metrics
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_path = os.path.join(
        "results",
        f"{folder_name}_predictions.npz"
    )

    np.savez(
        prediction_path,
        actual=actual,
        predicted=predictions
    )

    print(
        f"Predictions saved: "
        f"{prediction_path}"
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n")
    print("-" * 80)
    print(
        f"MODEL RESULTS - {stock_name}"
    )
    print("-" * 80)

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
        f"R2   : {r2:.4f}"
    )

    print(
        f"Epochs completed: "
        f"{epochs_completed}"
    )

    print("-" * 80)

    print(
        f"STATUS: "
        f"{stock_name} TRAINING COMPLETE"
    )

    return metrics


# ============================================================
# TRAIN ALL STOCKS
# ============================================================

def train_all_stocks():

    results = []

    successful = []

    failed = []

    print("\n")
    print("=" * 80)
    print(
        "          STARTING MULTI-STOCK TRAINING"
    )
    print("=" * 80)

    for stock_name, folder_name in STOCKS.items():

        # ----------------------------------------------------
        # TCS CONTROL
        # ----------------------------------------------------

        if (
            stock_name == "TCS"
            and not TRAIN_TCS
        ):

            print(
                "\nSkipping TCS "
                "(TRAIN_TCS = False)"
            )

            continue

        # ----------------------------------------------------
        # TRAIN STOCK
        # ----------------------------------------------------

        try:

            result = train_stock(
                stock_name,
                folder_name
            )

            results.append(
                result
            )

            if result["status"] == "trained":

                successful.append(
                    stock_name
                )

            else:

                failed.append(
                    stock_name
                )

        except Exception as error:

            print("\n")
            print(
                f"ERROR TRAINING "
                f"{stock_name}"
            )

            print(
                f"Error: {error}"
            )

            failed.append(
                stock_name
            )

    # ========================================================
    # SAVE MASTER METRICS
    # ========================================================

    master_metrics_path = os.path.join(
        "results",
        "all_stock_metrics.json"
    )

    with open(
        master_metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # ========================================================
    # SAVE MASTER CSV
    # ========================================================

    master_csv_path = os.path.join(
        "results",
        "all_stock_metrics.csv"
    )

    if results:

        fieldnames = [
            "stock",
            "status",
            "model",
            "epochs_completed",
            "MAE",
            "RMSE",
            "MAPE",
            "R2",
            "final_training_loss",
            "final_validation_loss"
        ]

        with open(
            master_csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                results
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "             TRAINING SUMMARY"
    )
    print("=" * 80)

    print(
        f"\nSuccessful: "
        f"{len(successful)}"
    )

    for stock in successful:

        print(
            f"   [OK] {stock}"
        )

    if failed:

        print(
            f"\nFailed: "
            f"{len(failed)}"
        )

        for stock in failed:

            print(
                f"   [FAILED] {stock}"
            )

    print("\n")
    print(
        "Models directory:"
    )

    print(
        "models/"
    )

    print("\n")
    print(
        "Metrics directory:"
    )

    print(
        "results/"
    )

    print("\n")
    print("=" * 80)
    print(
        "       MULTI-STOCK TRAINING FINISHED"
    )
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_all_stocks()