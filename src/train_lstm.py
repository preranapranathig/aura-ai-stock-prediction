import os
import json
import numpy as np
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# AURA AI — MULTI-STOCK LSTM TRAINING ENGINE
# ============================================================

print("=" * 80)
print("          AURA AI — MULTI-STOCK LSTM TRAINING ENGINE")
print("=" * 80)


# ============================================================
# CONFIGURATION
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

# ------------------------------------------------------------
# IMPORTANT
#
# Your original TCS model already works.
# Therefore we keep it untouched.
#
# Change this to True later if you want to retrain TCS.
# ------------------------------------------------------------

TRAIN_TCS = False


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

def build_model(input_shape):

    model = Sequential([

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

    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# ============================================================
# TRAIN ONE STOCK
# ============================================================

def train_stock(stock_name, folder_name):

    print("\n")
    print("=" * 80)
    print(f"TRAINING MODEL: {stock_name}")
    print("=" * 80)

    # --------------------------------------------------------
    # Existing TCS model protection
    # --------------------------------------------------------

    model_path = os.path.join(
        "models",
        f"{folder_name}_lstm_model.keras"
    )

    if (
        stock_name == "TCS"
        and not TRAIN_TCS
        and os.path.exists(model_path)
    ):

        print("\nTCS model already exists.")

        print(
            "Keeping the existing TCS model unchanged. ✅"
        )

        return {
            "stock": stock_name,
            "status": "existing",
            "model": model_path
        }

    # --------------------------------------------------------
    # Data paths
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
    # Check files
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

            print(
                f"\nERROR: Missing file:"
            )

            print(file_path)

            return {
                "stock": stock_name,
                "status": "failed"
            }

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading processed data...")

    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)

    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    scaler = joblib.load(
        scaler_path
    )

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"y_test:  {y_test.shape}"
    )

    # --------------------------------------------------------
    # Build model
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
    # Callbacks
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
    # Train
    # --------------------------------------------------------

    print("\n")
    print("-" * 80)

    print(
        f"Starting training for {stock_name}..."
    )

    print(
        f"Maximum epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
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
    # Save model
    # --------------------------------------------------------

    print("\nSaving model...")

    model.save(
        model_path
    )

    print(
        f"Model saved:"
    )

    print(model_path)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nEvaluating model...")

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    )

    predictions_scaled = predictions_scaled.reshape(-1, 1)

    actual_scaled = y_test.reshape(-1, 1)

    # --------------------------------------------------------
    # Convert predictions back to real price
    # --------------------------------------------------------

    predictions = scaler.inverse_transform(
        predictions_scaled
    ).flatten()

    actual = scaler.inverse_transform(
        actual_scaled
    ).flatten()

    # --------------------------------------------------------
    # Metrics
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

    # Prevent division by zero
    non_zero_mask = actual != 0

    if np.any(non_zero_mask):

        mape = np.mean(
            np.abs(
                (
                    actual[non_zero_mask]
                    -
                    predictions[non_zero_mask]
                )
                /
                actual[non_zero_mask]
            )
        ) * 100

    else:

        mape = 0.0

    r2 = r2_score(
        actual,
        predictions
    )

    # --------------------------------------------------------
    # Final training information
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
    # Metrics dictionary
    # --------------------------------------------------------

    metrics = {

        "stock": stock_name,

        "status": "trained",

        "model": model_path,

        "epochs_completed": epochs_completed,

        "MAE": float(mae),

        "RMSE": float(rmse),

        "MAPE": float(mape),

        "R2": float(r2),

        "final_training_loss":
            final_train_loss,

        "final_validation_loss":
            final_val_loss

    }

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = os.path.join(
        "results",
        f"{folder_name}_metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Save predictions
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

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n")
    print("-" * 80)

    print(
        f"MODEL RESULTS — {stock_name}"
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
        f"R²   : {r2:.4f}"
    )

    print(
        f"Epochs completed: {epochs_completed}"
    )

    print("-" * 80)

    print(
        f"STATUS: {stock_name} TRAINING COMPLETE ✅"
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

        try:

            result = train_stock(
                stock_name,
                folder_name
            )

            results.append(result)

            if result["status"] in [
                "trained",
                "existing"
            ]:

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
                f"ERROR TRAINING {stock_name}"
            )

            print(error)

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
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
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
        f"{len(successful)}/{len(STOCKS)}"
    )

    for stock in successful:

        print(
            f"   ✓ {stock}"
        )

    if failed:

        print(
            f"\nFailed: "
            f"{len(failed)}/{len(STOCKS)}"
        )

        for stock in failed:

            print(
                f"   ✗ {stock}"
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
        "       MULTI-STOCK TRAINING FINISHED 🚀"
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_all_stocks()