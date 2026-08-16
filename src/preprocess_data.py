import os
import joblib
import numpy as np
import pandas as pd

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
    "Tesla": "tesla.csv",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_folder_name(stock_name):
    """
    Convert stock name into a safe folder name.
    Example:
        HDFC Bank -> hdfc_bank
    """
    return stock_name.lower().replace(" ", "_")


# ============================================================
# PREPROCESS ONE STOCK
# ============================================================

def preprocess_stock(stock_name):
    """
    Preprocess one stock dataset.

    Features:
        Open
        High
        Low
        Close
        Volume

    Target:
        Close
    """

    # --------------------------------------------------------
    # Check stock name
    # --------------------------------------------------------

    if stock_name not in SUPPORTED_STOCKS:
        print(f"\nERROR: Unknown stock: {stock_name}")
        return False

    filename = SUPPORTED_STOCKS[stock_name]

    # --------------------------------------------------------
    # Input and output paths
    # --------------------------------------------------------

    input_file = os.path.join(
        "data",
        "stocks",
        filename,
    )

    output_folder = os.path.join(
        "data",
        "processed",
        safe_folder_name(stock_name),
    )

    os.makedirs(
        output_folder,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print(f"PREPROCESSING: {stock_name}")
    print("=" * 75)

    print("\nInput file:")
    print(input_file)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    if not os.path.exists(input_file):
        print("\nERROR: File not found:")
        print(input_file)
        return False

    df = pd.read_csv(input_file)

    print(
        f"\nOriginal dataset shape: {df.shape}"
    )

    # ========================================================
    # 2. CLEAN COLUMN NAMES
    # ========================================================

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # ========================================================
    # 3. CHECK DATE COLUMN
    # ========================================================

    if "Date" not in df.columns:
        print("\nERROR: Date column not found.")
        return False

    # ========================================================
    # 4. CONVERT DATE
    # ========================================================

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

    # ========================================================
    # 5. SORT BY DATE
    # ========================================================

    df = df.sort_values(
        "Date"
    )

    # ========================================================
    # 6. CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
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
    # 7. KEEP REQUIRED COLUMNS
    # ========================================================

    df = df[
        required_columns
    ]

    # ========================================================
    # 8. CONVERT NUMERIC COLUMNS
    # ========================================================

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ========================================================
    # 9. HANDLE MISSING VALUES
    # ========================================================

    print("\nMissing values BEFORE cleaning:")

    print(
        df[
            numeric_columns
        ].isna().sum()
    )

    # Forward fill
    df[numeric_columns] = (
        df[numeric_columns]
        .ffill()
    )

    # Backward fill remaining values
    df[numeric_columns] = (
        df[numeric_columns]
        .bfill()
    )

    # Remove rows with invalid dates
    df = df.dropna(
        subset=["Date"]
    )

    print("\nMissing values AFTER cleaning:")

    print(
        df[
            numeric_columns
        ].isna().sum()
    )

    # --------------------------------------------------------
    # Check whether numeric NaNs still exist
    # --------------------------------------------------------

    remaining_missing = (
        df[numeric_columns]
        .isna()
        .sum()
        .sum()
    )

    if remaining_missing > 0:

        print(
            "\nERROR: Missing numeric values "
            "still remain after cleaning."
        )

        return False

    # ========================================================
    # 10. REMOVE DUPLICATE DATES
    # ========================================================

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["Date"],
        keep="first",
    )

    duplicates_removed = (
        before_duplicates
        - len(df)
    )

    print(
        f"\nDuplicate dates removed: "
        f"{duplicates_removed}"
    )

    # ========================================================
    # 11. RESET INDEX
    # ========================================================

    df = df.reset_index(
        drop=True
    )

    print(
        f"Clean dataset shape: "
        f"{df.shape}"
    )

    # ========================================================
    # 12. CHECK DATA SIZE
    # ========================================================

    if len(df) <= SEQUENCE_LENGTH + 20:

        print(
            "\nERROR: Not enough historical data "
            "for LSTM training."
        )

        return False

    # ========================================================
    # 13. DISPLAY DATE RANGE
    # ========================================================

    first_date = (
        df["Date"]
        .iloc[0]
        .strftime("%Y-%m-%d")
    )

    last_date = (
        df["Date"]
        .iloc[-1]
        .strftime("%Y-%m-%d")
    )

    print("\nHistorical period:")
    print(first_date)
    print("to")
    print(last_date)

    # ========================================================
    # 14. SELECT OHLCV FEATURES
    # ========================================================

    feature_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    target_column = "Close"

    # Input features
    features = (
        df[
            feature_columns
        ].values
    )

    # Prediction target
    target = (
        df[
            [target_column]
        ].values
    )

    print("\nFeatures used:")
    print(feature_columns)

    print("\nPrediction target:")
    print(target_column)

    # ========================================================
    # 15. TRAIN / TEST SPLIT
    # ========================================================

    split_index = int(
        len(features)
        * TRAIN_RATIO
    )

    train_features = (
        features[:split_index]
    )

    test_features = (
        features[split_index:]
    )

    train_target = (
        target[:split_index]
    )

    test_target = (
        target[split_index:]
    )

    print("\nTRAIN / TEST SPLIT")

    print(
        f"Total samples: "
        f"{len(features)}"
    )

    print(
        f"Training samples: "
        f"{len(train_features)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_features)}"
    )

    # ========================================================
    # 16. FEATURE SCALER
    # ========================================================

    feature_scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    feature_scaler.fit(
        train_features
    )

    # ========================================================
    # 17. TARGET SCALER
    # ========================================================

    target_scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    target_scaler.fit(
        train_target
    )

    # ========================================================
    # 18. SCALE FEATURES
    # ========================================================

    train_features_scaled = (
        feature_scaler.transform(
            train_features
        )
    )

    test_features_scaled = (
        feature_scaler.transform(
            test_features
        )
    )

    # ========================================================
    # 19. SCALE TARGET
    # ========================================================

    train_target_scaled = (
        target_scaler.transform(
            train_target
        )
    )

    test_target_scaled = (
        target_scaler.transform(
            test_target
        )
    )

    # ========================================================
    # 20. CREATE TRAINING SEQUENCES
    # ========================================================

    X_train = []
    y_train = []

    for i in range(
        SEQUENCE_LENGTH,
        len(train_features_scaled),
    ):

        # Previous 60 days
        X_train.append(
            train_features_scaled[
                i - SEQUENCE_LENGTH:i
            ]
        )

        # Current day's Close price
        y_train.append(
            train_target_scaled[
                i,
                0
            ]
        )

    X_train = np.array(
        X_train
    )

    y_train = np.array(
        y_train
    )

    # ========================================================
    # 21. CREATE TEST SEQUENCES
    # ========================================================

    # We need the final 60 training days as context
    # for predicting the first test day.

    test_input_features = np.concatenate(
        [
            train_features_scaled[
                -SEQUENCE_LENGTH:
            ],
            test_features_scaled,
        ],
        axis=0,
    )

    X_test = []
    y_test = []

    for i in range(
        SEQUENCE_LENGTH,
        len(test_input_features),
    ):

        # Corresponding test target index
        test_index = (
            i - SEQUENCE_LENGTH
        )

        # 60-day input sequence
        X_test.append(
            test_input_features[
                i - SEQUENCE_LENGTH:i
            ]
        )

        # Actual Close price
        y_test.append(
            test_target_scaled[
                test_index,
                0
            ]
        )

    X_test = np.array(
        X_test
    )

    y_test = np.array(
        y_test
    )

    # ========================================================
    # 22. VALIDATE DATA SHAPES
    # ========================================================

    expected_features = (
        len(feature_columns)
    )

    print("\nDATA SHAPE VALIDATION")

    print(
        f"Expected input features: "
        f"{expected_features}"
    )

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"y_train shape: "
        f"{y_train.shape}"
    )

    print(
        f"X_test shape: "
        f"{X_test.shape}"
    )

    print(
        f"y_test shape: "
        f"{y_test.shape}"
    )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if X_train.ndim != 3:

        raise ValueError(
            "X_train must be 3-dimensional."
        )

    if X_test.ndim != 3:

        raise ValueError(
            "X_test must be 3-dimensional."
        )

    # --------------------------------------------------------
    # Check sequence length
    # --------------------------------------------------------

    if X_train.shape[1] != SEQUENCE_LENGTH:

        raise ValueError(
            f"X_train sequence length should be "
            f"{SEQUENCE_LENGTH}, but got "
            f"{X_train.shape[1]}."
        )

    if X_test.shape[1] != SEQUENCE_LENGTH:

        raise ValueError(
            f"X_test sequence length should be "
            f"{SEQUENCE_LENGTH}, but got "
            f"{X_test.shape[1]}."
        )

    # --------------------------------------------------------
    # Check feature count
    # --------------------------------------------------------

    if X_train.shape[2] != expected_features:

        raise ValueError(
            f"X_train should contain "
            f"{expected_features} features, "
            f"but got {X_train.shape[2]}."
        )

    if X_test.shape[2] != expected_features:

        raise ValueError(
            f"X_test should contain "
            f"{expected_features} features, "
            f"but got {X_test.shape[2]}."
        )

    # --------------------------------------------------------
    # Check X / y lengths
    # --------------------------------------------------------

    if len(X_train) != len(y_train):

        raise ValueError(
            "X_train and y_train have "
            "different lengths."
        )

    if len(X_test) != len(y_test):

        raise ValueError(
            "X_test and y_test have "
            "different lengths."
        )

    print("\nShape validation: SUCCESS")

    # ========================================================
    # 23. SAVE TRAINING DATA
    # ========================================================

    np.save(
        os.path.join(
            output_folder,
            "X_train.npy",
        ),
        X_train,
    )

    np.save(
        os.path.join(
            output_folder,
            "X_test.npy",
        ),
        X_test,
    )

    np.save(
        os.path.join(
            output_folder,
            "y_train.npy",
        ),
        y_train,
    )

    np.save(
        os.path.join(
            output_folder,
            "y_test.npy",
        ),
        y_test,
    )

    # ========================================================
    # 24. SAVE FEATURE SCALER
    # ========================================================

    feature_scaler_path = os.path.join(
        output_folder,
        "feature_scaler.pkl",
    )

    joblib.dump(
        feature_scaler,
        feature_scaler_path,
    )

    # ========================================================
    # 25. SAVE TARGET SCALER
    # ========================================================

    # Keep the filename "scaler.pkl".
    #
    # This scaler is ONLY for the Close-price target.
    #
    # Existing prediction/evaluation code can continue
    # loading scaler.pkl to inverse-transform predictions.

    scaler_path = os.path.join(
        output_folder,
        "scaler.pkl",
    )

    joblib.dump(
        target_scaler,
        scaler_path,
    )

    # ========================================================
    # 26. SAVE CLEANED DATA
    # ========================================================

    cleaned_path = os.path.join(
        output_folder,
        "cleaned.csv",
    )

    df.to_csv(
        cleaned_path,
        index=False,
    )

    # ========================================================
    # 27. FINAL REPORT
    # ========================================================

    print("\n")
    print("-" * 75)

    print(
        "PREPROCESSING COMPLETE"
    )

    print("-" * 75)

    print(
        f"Stock: {stock_name}"
    )

    print(
        f"Features: {feature_columns}"
    )

    print(
        f"Target: {target_column}"
    )

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

    print(
        f"Feature scaler features: "
        f"{feature_scaler.n_features_in_}"
    )

    print(
        f"Target scaler features: "
        f"{target_scaler.n_features_in_}"
    )

    print("\nFiles saved to:")
    print(output_folder)

    print("\nSTATUS: SUCCESS")

    print("=" * 75)

    return True


# ============================================================
# PROCESS ALL STOCKS
# ============================================================

def preprocess_all_stocks():
    """
    Preprocess every stock in SUPPORTED_STOCKS.
    """

    print("\n")
    print("=" * 80)
    print(
        "      AURA AI - MULTI-STOCK DATA PREPROCESSOR"
    )
    print("=" * 80)

    successful = []
    failed = []

    # --------------------------------------------------------
    # Process each stock
    # --------------------------------------------------------

    for stock_name in SUPPORTED_STOCKS:

        try:

            result = preprocess_stock(
                stock_name
            )

            if result:

                successful.append(
                    stock_name
                )

            else:

                failed.append(
                    stock_name
                )

        except Exception as error:

            print(
                f"\nERROR processing "
                f"{stock_name}:"
            )

            print(error)

            failed.append(
                stock_name
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        "             PREPROCESSING SUMMARY"
    )
    print("=" * 80)

    print(
        f"\nSuccessful: "
        f"{len(successful)}/"
        f"{len(SUPPORTED_STOCKS)}"
    )

    for stock in successful:

        print(
            f"   [OK] {stock}"
        )

    if failed:

        print(
            f"\nFailed: "
            f"{len(failed)}/"
            f"{len(SUPPORTED_STOCKS)}"
        )

        for stock in failed:

            print(
                f"   [FAILED] {stock}"
            )

    print(
        "\n" + "=" * 80
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    preprocess_all_stocks()