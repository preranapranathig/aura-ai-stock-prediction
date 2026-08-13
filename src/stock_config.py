# ============================================================
# MULTI-STOCK CONFIGURATION
# AI Stock Prediction System
# ============================================================

STOCKS = {

    # -------------------------------
    # 🇮🇳 INDIAN STOCKS
    # -------------------------------

    "TCS": {
        "name": "Tata Consultancy Services",
        "ticker": "TCS.NS",
        "market": "India",
        "currency": "₹",
        "model": "models/tcs_lstm_model.keras",
       "scaler": "data/processed/tcs/scaler.pkl"
    },

    "Infosys": {
        "name": "Infosys Limited",
        "ticker": "INFY.NS",
        "market": "India",
        "currency": "₹",
        "model": "models/infosys_lstm_model.keras",
        "scaler": "data/processed/infosys/scaler.pkl"
    },

    "Reliance": {
        "name": "Reliance Industries",
        "ticker": "RELIANCE.NS",
        "market": "India",
        "currency": "₹",
        "model": "models/reliance_lstm_model.keras",
        "scaler": "data/processed/reliance/scaler.pkl"
    },

    "HDFC Bank": {
        "name": "HDFC Bank Limited",
        "ticker": "HDFCBANK.NS",
        "market": "India",
        "currency": "₹",
        "model": "models/hdfc_bank_lstm_model.keras",
        "scaler": "data/processed/hdfc_bank/scaler.pkl"
    },

    # -------------------------------
    # 🇺🇸 US STOCKS
    # -------------------------------

    "Apple": {
        "name": "Apple Inc.",
        "ticker": "AAPL",
        "market": "USA",
        "currency": "$",
        "model": "models/apple_lstm_model.keras",
        "scaler": "data/processed/apple/scaler.pkl"
    },

    "Microsoft": {
        "name": "Microsoft Corporation",
        "ticker": "MSFT",
        "market": "USA",
        "currency": "$",
        "model": "models/microsoft_lstm_model.keras",
        "scaler": "data/processed/microsoft/scaler.pkl"
    },

    "Tesla": {
        "name": "Tesla Inc.",
        "ticker": "TSLA",
        "market": "USA",
        "currency": "$",
        "model": "models/tesla_lstm_model.keras",
        "scaler": "data/processed/tesla/scaler.pkl"
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_stock(stock_name):
    """
    Return configuration for a selected stock.
    """

    if stock_name not in STOCKS:
        raise ValueError(f"Unknown stock: {stock_name}")

    return STOCKS[stock_name]


def get_stock_names():
    """
    Return all available stock names.
    """

    return list(STOCKS.keys())


def get_indian_stocks():
    """
    Return Indian stock names.
    """

    return [
        name
        for name, info in STOCKS.items()
        if info["market"] == "India"
    ]


def get_us_stocks():
    """
    Return US stock names.
    """

    return [
        name
        for name, info in STOCKS.items()
        if info["market"] == "USA"
    ]