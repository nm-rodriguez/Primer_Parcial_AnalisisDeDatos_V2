import pandas as pd
import numpy as np

# Load raw file
df_sample = pd.read_csv('rentals.txt', sep=';', nrows=100)
print("Columns:", df_sample.columns.tolist())
print("\nFirst 3 rows:")
print(df_sample.head(3).T)

# Load full file
df = pd.read_csv('rentals.txt', sep=';')
print(f"\nShape: {df.shape}")
print("\nData Types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nBasic description of numeric columns:")
print(df.describe().T)
