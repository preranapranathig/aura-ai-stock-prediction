import pandas as pd
import numpy as np
import os
import joblib
import sys

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# MULTI-STOCK PREPROCESSING ENGINE
# ============================================================

SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.80

SUPPORTED_STOCKS = {
    "TCS": "tcs.csv",
    "Infosys": "infosys.csv",
    "Reliance": "reliance.csv",
    "HDFC Bank": "hdfc_bank.csv",
    "Apple": "apple.csv",
    "Microsoft": "microsoft.csv",
    "Tesla": "tesla.csv"
}


# ============================================================
# HELPER
# ============================================================

def safe_folder_name(stock_name):
    return stock_name.lower().replace(" ", "_")


# ============================================================
# PREPROCESS ONE STOCK
# ============================================================

def preprocess_stock(stock_name):

    if stock_name not in SUPPORTED_STOCKS:
        print(f"\nERROR: Unknown stock: {stock_name}")
        return False

    filename = SUPPORTED_STOCKS[stock_name]

    input_file = os.path.join(
        "data",
        "stocks",
        filename
    )

    output_folder = os.path.join(
        "data",
        "processed",
        safe_folder_name(stock_name)
    )

    os.makedirs(output_folder, exist_ok=True)

    print("\n")
    print("=" * 75)
    print(f"PREPROCESSING: {stock_name}")
    print("=" * 75)

    print(f"\nInput file:")
    print(input_file)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    if not os.path.exists(input_file):

        print(f"\nERROR: File not found:")
        print(input_file)

        return False

    df = pd.read_csv(input_file)

    print(f"\nOriginal dataset shape: {df.shape}")

    # ========================================================
    # 2. CLEAN COLUMN NAMES
    # ========================================================

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # ========================================================
    # 3. CONVERT DATE
    # ========================================================

    if "Date" not in df.columns:

        print("\nERROR: Date column not found.")

        return False

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # ========================================================
    # 4. SORT BY DATE
    # ========================================================

    df = df.sort_values("Date")

    # ========================================================
    # 5. CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print("\nERROR: Missing columns:")
        print(missing_columns)

        return False

    # ========================================================
    # 6. KEEP REQUIRED COLUMNS
    # ========================================================

    df = df[required_columns]

    # ========================================================
    # 7. CONVERT NUMERIC DATA
    # ========================================================

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ========================================================
    # 8. REMOVE MISSING VALUES
    # ========================================================

    before_cleaning = len(df)

    df = df.dropna()

    after_cleaning = len(df)

    print(
        f"\nRemoved missing rows: "
        f"{before_cleaning - after_cleaning}"
    )

    # ========================================================
    # 9. REMOVE DUPLICATES
    # ========================================================

    df = df.drop_duplicates(
        subset=["Date"]
    )

    # ========================================================
    # 10. RESET INDEX
    # ========================================================

    df = df.reset_index(drop=True)

    print(
        f"Clean dataset shape: {df.shape}"
    )

    # ========================================================
    # 11. CHECK DATA SIZE
    # ========================================================

    if len(df) <= SEQUENCE_LENGTH + 20:

        print(
            "\nERROR: Not enough historical data "
            "for LSTM training."
        )

        return False

    # ========================================================
    # 12. DISPLAY DATE RANGE
    # ========================================================

    print("\nHistorical period:")

    print(
        df["Date"].iloc[0].strftime("%Y-%m-%d")
    )

    print("to")

    print(
        df["Date"].iloc[-1].strftime("%Y-%m-%d")
    )

    # ========================================================
    # 13. SELECT CLOSE PRICE
    # ========================================================

    prices = df["Close"].values.reshape(-1, 1)

    # ========================================================
    # 14. TRAIN / TEST SPLIT
    # ========================================================

    split_index = int(
        len(prices) * TRAIN_RATIO
    )

    train_prices = prices[:split_index]

    test_prices = prices[split_index:]

    print("\nTRAIN / TEST SPLIT")

    print(
        f"Total samples: {len(prices)}"
    )

    print(
        f"Training samples: {len(train_prices)}"
    )

    print(
        f"Testing samples: {len(test_prices)}"
    )

    # ========================================================
    # 15. FIT SCALER ONLY ON TRAINING DATA
    # ========================================================

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    scaler.fit(train_prices)

    # ========================================================
    # 16. SCALE DATA
    # ========================================================

    train_scaled = scaler.transform(
        train_prices
    )

    test_scaled = scaler.transform(
        test_prices
    )

    # ========================================================
    # 17. CREATE TRAINING SEQUENCES
    # ========================================================

    X_train = []
    y_train = []

    for i in range(
        SEQUENCE_LENGTH,
        len(train_scaled)
    ):

        X_train.append(
            train_scaled[
                i - SEQUENCE_LENGTH:i,
                0
            ]
        )

        y_train.append(
            train_scaled[i, 0]
        )

    X_train = np.array(X_train)

    y_train = np.array(y_train)

    # ========================================================
    # 18. CREATE TEST SEQUENCES
    # ========================================================

    test_input = np.concatenate(
        [
            train_scaled[-SEQUENCE_LENGTH:],
            test_scaled
        ]
    )

    X_test = []
    y_test = []

    for i in range(
        SEQUENCE_LENGTH,
        len(test_input)
    ):

        X_test.append(
            test_input[
                i - SEQUENCE_LENGTH:i,
                0
            ]
        )

        y_test.append(
            test_input[i, 0]
        )

    X_test = np.array(X_test)

    y_test = np.array(y_test)

    # ========================================================
    # 19. RESHAPE FOR LSTM
    # ========================================================

    X_train = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    X_test = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        1
    )

    # ========================================================
    # 20. SAVE TRAINING DATA
    # ========================================================

    np.save(
        os.path.join(
            output_folder,
            "X_train.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            output_folder,
            "X_test.npy"
        ),
        X_test
    )

    np.save(
        os.path.join(
            output_folder,
            "y_train.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            output_folder,
            "y_test.npy"
        ),
        y_test
    )

    # ========================================================
    # 21. SAVE SCALER
    # ========================================================

    scaler_path = os.path.join(
        output_folder,
        "scaler.pkl"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    # ========================================================
    # 22. SAVE CLEAN DATA
    # ========================================================

    cleaned_path = os.path.join(
        output_folder,
        "cleaned.csv"
    )

    df.to_csv(
        cleaned_path,
        index=False
    )

    # ========================================================
    # 23. FINAL REPORT
    # ========================================================

    print("\n" + "-" * 75)

    print(
        f"X_train shape: {X_train.shape}"
    )

    print(
        f"y_train shape: {y_train.shape}"
    )

    print(
        f"X_test shape:  {X_test.shape}"
    )

    print(
        f"y_test shape:  {y_test.shape}"
    )

    print("\nFiles saved to:")

    print(output_folder)

    print("\nSTATUS: SUCCESS ✅")

    print("=" * 75)

    return True


# ============================================================
# PROCESS ALL STOCKS
# ============================================================

def preprocess_all_stocks():

    print("\n")
    print("=" * 80)
    print("      AURA AI — MULTI-STOCK DATA PREPROCESSOR")
    print("=" * 80)

    successful = []
    failed = []

    for stock_name in SUPPORTED_STOCKS:

        try:

            result = preprocess_stock(
                stock_name
            )

            if result:
                successful.append(stock_name)
            else:
                failed.append(stock_name)

        except Exception as error:

            print(
                f"\nERROR processing {stock_name}:"
            )

            print(error)

            failed.append(stock_name)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("             PREPROCESSING SUMMARY")
    print("=" * 80)

    print(
        f"\nSuccessful: "
        f"{len(successful)}/{len(SUPPORTED_STOCKS)}"
    )

    for stock in successful:

        print(f"   ✓ {stock}")

    if failed:

        print(
            f"\nFailed: "
            f"{len(failed)}/{len(SUPPORTED_STOCKS)}"
        )

        for stock in failed:

            print(f"   ✗ {stock}")

    print("\n" + "=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    preprocess_all_stocks()