import pandas as pd

# Load the downloaded dataset
file_path = "data/tcs_stock_data.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("TCS STOCK DATASET INSPECTION")
print("=" * 60)

print("\n1. Dataset shape:")
print(df.shape)

print("\n2. Columns:")
print(df.columns.tolist())

print("\n3. First 5 rows:")
print(df.head())

print("\n4. Last 5 rows:")
print(df.tail())

print("\n5. Missing values:")
print(df.isnull().sum())

print("\n6. Data types:")
print(df.dtypes)

print("\n7. Basic statistics:")
print(df.describe())