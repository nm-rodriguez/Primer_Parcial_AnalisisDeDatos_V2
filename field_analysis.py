import pandas as pd
import numpy as np

df = pd.read_csv('rentals.txt', sep=';')
df['monthly_rent_ars_num'] = df['monthly_rent_ars'].str.replace('$ ', '', regex=False).str.replace('.', '', regex=False).astype(float)
df['monthly_rent_rub_num'] = df['monthly_rent_rub'].str.replace('$ ', '', regex=False).str.replace('.', '', regex=False).astype(float)
df['date_posted'] = pd.to_datetime(df['date_posted'], format='%d/%m/%Y')

# Real values scaling
df['total_area_real'] = df['total_area'] / 10.0 # m2
df['to_center_km_real'] = df['to_center_km'] / 100.0 # km

print("=== TEMPORAL ANALYSIS ===")
df['year_posted'] = df['date_posted'].dt.year
print(df.groupby('year_posted')['monthly_rent_ars_num'].agg(['count', 'mean', 'median']))

print("\n=== TOP 10 EXPENSIVE DISTRICTS ===")
dist_stats = df.groupby('district')['monthly_rent_ars_num'].agg(['count', 'mean', 'median'])
print(dist_stats.sort_values('mean', ascending=False).head(10))

print("\n=== TOP 10 CHEAPEST DISTRICTS ===")
print(dist_stats.sort_values('mean', ascending=True).head(10))

print("\n=== DISTRICT VS OKRUG VARIANCE EXPLAINED ===")
# One-way ANOVA or R2 from groupby mean
okrug_pred = df.groupby('okrug')['monthly_rent_ars_num'].transform('mean')
district_pred = df.groupby('district')['monthly_rent_ars_num'].transform('mean')
ss_total = ((df['monthly_rent_ars_num'] - df['monthly_rent_ars_num'].mean()) ** 2).sum()
ss_res_okrug = ((df['monthly_rent_ars_num'] - okrug_pred) ** 2).sum()
ss_res_dist = ((df['monthly_rent_ars_num'] - district_pred) ** 2).sum()
r2_okrug = 1 - (ss_res_okrug / ss_total)
r2_district = 1 - (ss_res_dist / ss_total)
print(f"Variance explained (R2) by Okrug alone: {r2_okrug:.4f}")
print(f"Variance explained (R2) by District alone: {r2_district:.4f}")

print("\n=== FLOOR RELATIONSHIP ===")
df['floor_ratio'] = df['floor'] / df['total_floors']
df['is_first_floor'] = (df['floor'] == 1)
df['is_last_floor'] = (df['floor'] == df['total_floors'])
print("First floor avg price:", df.groupby('is_first_floor')['monthly_rent_ars_num'].mean())
print("Last floor avg price:", df.groupby('is_last_floor')['monthly_rent_ars_num'].mean())
print("Correlation floor_ratio with price:", df['floor_ratio'].corr(df['monthly_rent_ars_num']))

print("\n=== METRO DISTANCE ANALYSIS ===")
print("metro_distance_min describe:")
print(df['metro_distance_min'].describe())
# Is metro_distance_min also scaled by 10? E.g. min 0, 25% 33, median 103, max 966.
# If 966 is minutes, that is 16 hours! If scaled by 10, 96.6 minutes (1.5 hours) or walking distance in decaminutes?
print("Correlation metro_distance_min with to_center_km:", df['metro_distance_min'].corr(df['to_center_km']))
print("Correlation metro_distance_min with monthly_rent_ars_num:", df['metro_distance_min'].corr(df['monthly_rent_ars_num']))
