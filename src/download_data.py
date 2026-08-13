import os
import time
import yfinance as yf
import pandas as pd

from stock_config import STOCKS


# ============================================================
# AURA AI - MULTI-STOCK MARKET DATA ENGINE
# ============================================================

DATA_FOLDER = "data/stocks"

# Historical period used for model training
HISTORICAL_PERIOD = "5y"

# Daily candles
INTERVAL = "1d"


def download_stock_data(stock_name):
    """
    Download fresh historical market data for one stock.
    Existing CSV is replaced with the newest downloaded data.
    """

    if stock_name not in STOCKS:
        print(f"❌ ERROR: {stock_name} is not configured.")
        return False

    config = STOCKS[stock_name]

    ticker = config["ticker"]
    company_name = config["name"]
    market = config["market"]
    currency = config["currency"]

    print("\n" + "=" * 70)
    print(f"📡 AURA AI MARKET DATA ENGINE")
    print("=" * 70)
    print(f"Stock       : {stock_name}")
    print(f"Company     : {company_name}")
    print(f"Ticker      : {ticker}")
    print(f"Market      : {market}")
    print(f"Currency    : {currency}")
    print(f"Period      : {HISTORICAL_PERIOD}")
    print(f"Interval    : {INTERVAL}")
    print("=" * 70)

    try:
        # ----------------------------------------------------
        # DOWNLOAD FROM YAHOO FINANCE
        # ----------------------------------------------------

        data = yf.download(
            ticker,
            period=HISTORICAL_PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False
        )

        # ----------------------------------------------------
        # CHECK DOWNLOAD
        # ----------------------------------------------------

        if data is None or data.empty:
            print(f"❌ No data received for {stock_name}")
            return False

        # ----------------------------------------------------
        # HANDLE YFINANCE MULTI-LEVEL COLUMNS
        # ----------------------------------------------------

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # ----------------------------------------------------
        # CLEAN DATA
        # ----------------------------------------------------

        data = data.copy()

        data = data.dropna()

        # Remove duplicate dates
        data = data[~data.index.duplicated(keep="last")]

        # Make sure data is sorted chronologically
        data = data.sort_index()

        # ----------------------------------------------------
        # REQUIRED COLUMNS
        # ----------------------------------------------------

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            print(
                f"❌ Missing columns for {stock_name}: "
                f"{missing_columns}"
            )
            return False

        # ----------------------------------------------------
        # CREATE DATA DIRECTORY
        # ----------------------------------------------------

        os.makedirs(DATA_FOLDER, exist_ok=True)

        # ----------------------------------------------------
        # SAFE FILE NAME
        # ----------------------------------------------------

        filename = stock_name.lower().replace(" ", "_")

        output_file = os.path.join(
            DATA_FOLDER,
            f"{filename}.csv"
        )

        # ----------------------------------------------------
        # SAVE FRESH DATA
        # ----------------------------------------------------

        data.to_csv(output_file)

        # ----------------------------------------------------
        # DATA SUMMARY
        # ----------------------------------------------------

        first_date = data.index[0]
        last_date = data.index[-1]

        latest_close = float(data["Close"].iloc[-1])

        print("\n✅ DOWNLOAD SUCCESSFUL")
        print("-" * 70)
        print(f"Saved file       : {output_file}")
        print(f"Rows             : {len(data):,}")
        print(f"First date       : {first_date}")
        print(f"Latest date      : {last_date}")
        print(f"Latest close     : {latest_close:,.2f}")
        print("-" * 70)

        print("\n📊 Latest 3 trading days:")
        print(data.tail(3).to_string())

        return True

    except Exception as error:

        print("\n❌ DOWNLOAD FAILED")
        print(f"Stock: {stock_name}")
        print(f"Reason: {error}")

        return False


# ============================================================
# DOWNLOAD ALL STOCKS
# ============================================================

def download_all_stocks():
    """
    Download fresh historical data for all configured stocks.
    """

    print("\n")
    print("=" * 75)
    print("       🚀 AURA AI - MARKET DATA ENGINE")
    print("=" * 75)
    print("       Fresh historical market data")
    print("=" * 75)

    successful = []
    failed = []

    stock_names = list(STOCKS.keys())

    print(f"\nStocks configured: {len(stock_names)}")
    print(f"Stocks: {', '.join(stock_names)}")

    # --------------------------------------------------------
    # DOWNLOAD EACH STOCK
    # --------------------------------------------------------

    for index, stock_name in enumerate(stock_names, start=1):

        print(
            f"\n\n[{index}/{len(stock_names)}] "
            f"Processing {stock_name}..."
        )

        success = download_stock_data(stock_name)

        if success:
            successful.append(stock_name)
        else:
            failed.append(stock_name)

        # Small pause between requests
        time.sleep(1)

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 75)
    print("              📈 DATA ENGINE REPORT")
    print("=" * 75)

    print(f"\n✅ Successful: {len(successful)}/{len(stock_names)}")

    for stock in successful:
        print(f"   ✓ {stock}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(stock_names)}")

        for stock in failed:
            print(f"   ✗ {stock}")

    print("\n" + "=" * 75)

    if not failed:
        print("🎉 ALL 7 STOCK DATASETS UPDATED SUCCESSFULLY!")
    else:
        print("⚠️ SOME DATASETS COULD NOT BE UPDATED.")

    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    download_all_stocks()