# ============================================================
# AURA AI - FINAL MODEL EVALUATION & COMPARISON
# ============================================================

import os
import pandas as pd

RESULTS_DIR = "results"

FILES = {
    "LSTM": "all_stock_metrics.csv",
    "BiLSTM": "all_stock_bilstm_metrics.csv",
    "LSTM + Attention": "all_stock_lstm_attention_metrics.csv",
    "BiLSTM + Attention": "all_stock_bilstm_attention_metrics.csv",
}

# ============================================================
# CANONICAL STOCK NAMES
# ============================================================

STOCK_MAP = {
    "tcs": "TCS",
    "infosys": "Infosys",
    "reliance": "Reliance",
    "hdfc bank": "HDFC Bank",
    "hdfc_bank": "HDFC Bank",
    "apple": "Apple",
    "microsoft": "Microsoft",
    "tesla": "Tesla",
}

EXPECTED_STOCKS = [
    "TCS",
    "Infosys",
    "Reliance",
    "HDFC Bank",
    "Apple",
    "Microsoft",
    "Tesla",
]


def normalize_stock(value):

    if pd.isna(value):
        return None

    key = str(value).strip().lower()

    return STOCK_MAP.get(key, str(value).strip())


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("AURA AI - FINAL MODEL COMPARISON")
print("=" * 80)

all_results = []


# ============================================================
# LOAD MODEL METRICS
# ============================================================

for model_name, filename in FILES.items():

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    print("\n" + "-" * 80)
    print(f"Checking: {filename}")

    if not os.path.exists(path):

        print(
            f"[MISSING] {filename}"
        )

        continue

    try:

        df = pd.read_csv(path)

        print(
            f"[FOUND] {filename} -> {len(df)} rows"
        )

        # ----------------------------------------------------
        # Normalize column names
        # ----------------------------------------------------

        rename_map = {}

        for col in df.columns:

            lower = str(col).strip().lower()

            if lower == "stock":
                rename_map[col] = "stock"

            elif lower == "mae":
                rename_map[col] = "MAE"

            elif lower == "rmse":
                rename_map[col] = "RMSE"

            elif lower == "mape":
                rename_map[col] = "MAPE"

            elif lower in ["r2", "r²"]:
                rename_map[col] = "R2"

        df = df.rename(
            columns=rename_map
        )

        required = [
            "stock",
            "MAE",
            "RMSE",
            "MAPE",
            "R2"
        ]

        missing = [
            col
            for col in required
            if col not in df.columns
        ]

        if missing:

            print(
                f"[ERROR] Missing columns: {missing}"
            )

            continue

        # ----------------------------------------------------
        # Normalize stock names
        # ----------------------------------------------------

        df["stock"] = df["stock"].apply(
            normalize_stock
        )

        # ----------------------------------------------------
        # Convert metrics to numeric
        # ----------------------------------------------------

        for column in [
            "MAE",
            "RMSE",
            "MAPE",
            "R2"
        ]:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # Remove invalid rows
        # ----------------------------------------------------

        df = df.dropna(
            subset=[
                "stock",
                "MAE",
                "RMSE",
                "MAPE",
                "R2"
            ]
        )

        # ----------------------------------------------------
        # Keep only required columns
        # ----------------------------------------------------

        df = df[
            [
                "stock",
                "MAE",
                "RMSE",
                "MAPE",
                "R2"
            ]
        ].copy()

        # ----------------------------------------------------
        # Add model name
        # ----------------------------------------------------

        df["comparison_model"] = model_name

        # ----------------------------------------------------
        # Remove duplicate stock rows INSIDE this model
        # ----------------------------------------------------
        #
        # If the same model accidentally contains:
        #
        # TCS
        # tcs
        #
        # they are now both normalized to TCS.
        #
        # We keep the first valid result.
        # ----------------------------------------------------

        before = len(df)

        df = df.drop_duplicates(
            subset=[
                "stock",
                "comparison_model"
            ],
            keep="first"
        )

        after = len(df)

        if before != after:

            print(
                f"[FIXED] Removed "
                f"{before - after} duplicate stock rows"
            )

        all_results.append(df)

        print(
            f"[OK] {model_name:<24} "
            f"{len(df)} unique stock results"
        )

        print(
            "Stocks:",
            ", ".join(
                df["stock"].tolist()
            )
        )

    except Exception as e:

        print(
            f"[ERROR] {filename}: {e}"
        )


# ============================================================
# CHECK LOADED MODELS
# ============================================================

print("\n")
print("=" * 80)
print("MODEL FILE STATUS")
print("=" * 80)

loaded_models = []

for df in all_results:

    model = df["comparison_model"].iloc[0]

    loaded_models.append(model)

for model_name in FILES:

    if model_name in loaded_models:

        print(
            f"[OK]      {model_name}"
        )

    else:

        print(
            f"[MISSING] {model_name}"
        )


# ============================================================
# COMBINE ALL RESULTS
# ============================================================

if not all_results:

    print(
        "\nERROR: No metric files could be loaded."
    )

    raise SystemExit(1)


comparison = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# FINAL CLEANUP
# ============================================================

comparison["stock"] = comparison[
    "stock"
].apply(
    normalize_stock
)


# Keep only the 7 expected companies

comparison = comparison[
    comparison["stock"].isin(
        EXPECTED_STOCKS
    )
].copy()


# ============================================================
# SORT STOCKS
# ============================================================

stock_order = {
    stock: index
    for index, stock
    in enumerate(EXPECTED_STOCKS)
}

comparison["_stock_order"] = (
    comparison["stock"]
    .map(stock_order)
)


# ============================================================
# METRIC RANKING
# ============================================================

print("\n")
print("=" * 80)
print("CALCULATING MODEL RANKINGS")
print("=" * 80)


# Lower is better

comparison["MAE_rank"] = (
    comparison
    .groupby("stock")["MAE"]
    .rank(
        method="min",
        ascending=True
    )
)


comparison["RMSE_rank"] = (
    comparison
    .groupby("stock")["RMSE"]
    .rank(
        method="min",
        ascending=True
    )
)


comparison["MAPE_rank"] = (
    comparison
    .groupby("stock")["MAPE"]
    .rank(
        method="min",
        ascending=True
    )
)


# Higher is better

comparison["R2_rank"] = (
    comparison
    .groupby("stock")["R2"]
    .rank(
        method="min",
        ascending=False
    )
)


# ============================================================
# COMBINED SCORE
# ============================================================

comparison["combined_rank"] = (
    comparison["MAE_rank"]
    + comparison["RMSE_rank"]
    + comparison["MAPE_rank"]
    + comparison["R2_rank"]
)


# ============================================================
# SAVE MASTER COMPARISON
# ============================================================

master_path = os.path.join(
    RESULTS_DIR,
    "final_model_comparison.csv"
)


comparison.sort_values(
    [
        "_stock_order",
        "combined_rank"
    ]
).drop(
    columns=["_stock_order"]
).to_csv(
    master_path,
    index=False
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_models = (
    comparison
    .sort_values(
        [
            "stock",
            "combined_rank",
            "MAE",
            "RMSE",
            "MAPE"
        ]
    )
    .groupby(
        "stock",
        as_index=False
    )
    .first()
)


# ============================================================
# SAVE BEST MODELS
# ============================================================

best_path = os.path.join(
    RESULTS_DIR,
    "best_models_by_stock.csv"
)


best_models[
    [
        "stock",
        "comparison_model",
        "MAE",
        "RMSE",
        "MAPE",
        "R2",
        "combined_rank"
    ]
].to_csv(
    best_path,
    index=False
)


# ============================================================
# DISPLAY MASTER COMPARISON
# ============================================================

print("\n")
print("=" * 80)
print("MASTER MODEL COMPARISON")
print("=" * 80)

display_columns = [
    "stock",
    "comparison_model",
    "MAE",
    "RMSE",
    "MAPE",
    "R2",
    "combined_rank"
]

print(
    comparison.sort_values(
        [
            "_stock_order",
            "combined_rank"
        ]
    )[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# DISPLAY BEST MODEL
# ============================================================

print("\n")
print("=" * 80)
print("BEST MODEL FOR EACH STOCK")
print("=" * 80)

print(
    best_models[
        [
            "stock",
            "comparison_model",
            "MAE",
            "RMSE",
            "MAPE",
            "R2",
            "combined_rank"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# VERIFY EXACTLY 7 STOCKS
# ============================================================

print("\n")
print("=" * 80)
print("STOCK COVERAGE CHECK")
print("=" * 80)

best_stock_set = set(
    best_models["stock"]
)

missing_stocks = [
    stock
    for stock in EXPECTED_STOCKS
    if stock not in best_stock_set
]

extra_stocks = [
    stock
    for stock in best_stock_set
    if stock not in EXPECTED_STOCKS
]


print(
    f"Unique stocks evaluated : "
    f"{comparison['stock'].nunique()}"
)

print(
    f"Best-model rows         : "
    f"{len(best_models)}"
)


if missing_stocks:

    print(
        "[WARNING] Missing stocks:",
        missing_stocks
    )

else:

    print(
        "[OK] All 7 companies have a best model."
    )


if extra_stocks:

    print(
        "[WARNING] Unexpected stocks:",
        extra_stocks
    )


# ============================================================
# MODEL WIN COUNT
# ============================================================

wins = (
    best_models[
        "comparison_model"
    ]
    .value_counts()
)


print("\n")
print("=" * 80)
print("MODEL WIN COUNT")
print("=" * 80)

print(
    wins.to_string()
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 80)
print("EVALUATION COMPLETED")
print("=" * 80)

print(
    f"\nMaster comparison:"
    f"\n{master_path}"
)

print(
    f"\nBest models:"
    f"\n{best_path}"
)

print("\n")
print(
    "IMPORTANT:"
)
print(
    "The script does NOT retrain any model."
)
print(
    "It only compares existing trained metrics."
)

print("=" * 80)