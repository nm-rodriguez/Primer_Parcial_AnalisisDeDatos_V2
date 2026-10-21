import pandas as pd
import numpy as np

df = pd.read_csv('rentals.txt', sep=';')

# Convert price
df['monthly_rent_rub_num'] = df['monthly_rent_rub'].str.replace('$ ', '', regex=False).str.replace('.', '', regex=False).astype(float)
df['monthly_rent_ars_num'] = df['monthly_rent_ars'].str.replace('$ ', '', regex=False).str.replace('.', '', regex=False).astype(float)

# Verify ARS = RUB * 18
print("Differences between ARS and RUB * 18:", (df['monthly_rent_ars_num'] - df['monthly_rent_rub_num'] * 18).abs().max())

# Check duplicates
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate IDs:", df['id'].duplicated().sum())

# Consistency checks
print("\n--- INTEGRITY CHECKS ---")
invalid_floors = df[df['floor'] > df['total_floors']]
print(f"Cases where floor > total_floors: {len(invalid_floors)}")

invalid_years = df[(df['building_year'] < 1800) | (df['building_year'] > 2026)]
print(f"Cases with invalid building_year (<1800 or >2026): {len(invalid_years)}")

invalid_area = df[df['total_area'] <= 0]
print(f"Cases with total_area <= 0: {len(invalid_area)}")

invalid_price = df[df['monthly_rent_ars_num'] <= 0]
print(f"Cases with monthly_rent_ars <= 0: {len(invalid_price)}")

invalid_rooms = df[df['rooms'] <= 0]
print(f"Cases with rooms <= 0: {len(invalid_rooms)}")

# Statistical summary of target variable
print("\n--- TARGET VARIABLE (monthly_rent_ars) ---")
target = df['monthly_rent_ars_num']
desc = target.describe()
print(desc)
q1 = desc['25%']
q3 = desc['75%']
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers = target[(target < lower_fence) | (target > upper_fence)]
print(f"IQR: {iqr}")
print(f"Lower fence: {lower_fence}, Upper fence: {upper_fence}")
print(f"Outliers count (Tukey IQR): {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
print(f"Skewness: {target.skew():.2f}")
print(f"Kurtosis: {target.kurt():.2f}")

# Target correlation with numeric features (excluding monthly_rent_rub and rent_per_sqm to avoid leakage)
numeric_cols = ['total_area', 'rooms', 'floor', 'total_floors', 'building_year', 'metro_distance_min', 'to_center_km', 'deposit_months']
print("\n--- CORRELATION WITH NUMERIC FEATURES ---")
corrs = df[numeric_cols].apply(lambda s: s.corr(df['monthly_rent_ars_num']))
print(corrs.sort_values(ascending=False))

# Full correlation matrix among numeric features
print("\n--- NUMERIC CORRELATION MATRIX ---")
corr_matrix = df[numeric_cols + ['monthly_rent_ars_num']].corr()
print(corr_matrix.round(3))

# Categorical breakdowns (Mean, Median, Std, Count)
print("\n--- PRICE BY OKRUG ---")
okrug_stats = df.groupby('okrug')['monthly_rent_ars_num'].agg(['count', 'mean', 'median', 'std']).sort_values('mean', ascending=False)
print(okrug_stats.round(1))

print("\n--- PRICE BY BUILDING TYPE ---")
bt_stats = df.groupby('building_type')['monthly_rent_ars_num'].agg(['count', 'mean', 'median', 'std']).sort_values('mean', ascending=False)
print(bt_stats.round(1))

print("\n--- PRICE BY RENOVATION ---")
renov_stats = df.groupby('renovation')['monthly_rent_ars_num'].agg(['count', 'mean', 'median', 'std']).sort_values('mean', ascending=False)
print(renov_stats.round(1))

print("\n--- PRICE BY ROOMS ---")
room_stats = df.groupby('rooms')['monthly_rent_ars_num'].agg(['count', 'mean', 'median', 'std']).sort_values('mean', ascending=False)
print(room_stats.round(1))

print("\n--- PRICE BY FURNISHED ---")
print(df.groupby('furnished')['monthly_rent_ars_num'].agg(['count', 'mean', 'median']).round(1))

print("\n--- PRICE BY PETS ALLOWED ---")
print(df.groupby('pets_allowed')['monthly_rent_ars_num'].agg(['count', 'mean', 'median']).round(1))

print("\n--- PRICE BY DEPOSIT MONTHS ---")
print(df.groupby('deposit_months')['monthly_rent_ars_num'].agg(['count', 'mean', 'median']).round(1))
