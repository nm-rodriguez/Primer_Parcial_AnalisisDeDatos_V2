import pandas as pd
import numpy as np

df = pd.read_csv('rentals.txt', sep=';')
df['monthly_rent_ars_num'] = df['monthly_rent_ars'].str.replace('$ ', '', regex=False).str.replace('.', '', regex=False).astype(float)
df['total_area_m2'] = df['total_area'] / 10.0
df['to_center_km_real'] = df['to_center_km'] / 100.0
df['metro_distance_min_real'] = df['metro_distance_min'] # let's check
df['price_per_sqm_ars'] = df['monthly_rent_ars_num'] / df['total_area_m2']

# Let's segment by distance to center:
# Central: to_center_km_real <= 5 km
# Middle ring: 5 < to_center_km_real <= 15 km
# Periphery/Suburban: to_center_km_real > 15 km
df['zone'] = pd.cut(df['to_center_km_real'], bins=[0, 5, 15, 100], labels=['Central (<5km)', 'Intermedia (5-15km)', 'Periferia (>15km)'])

print("=== DISTRIBUCIÓN POR ZONA ===")
zone_stats = df.groupby('zone', observed=False).agg(
    n=('monthly_rent_ars_num', 'count'),
    precio_medio=('monthly_rent_ars_num', 'mean'),
    precio_mediano=('monthly_rent_ars_num', 'median'),
    area_media=('total_area_m2', 'mean'),
    precio_m2_medio=('price_per_sqm_ars', 'mean'),
    corr_area_precio=('total_area_m2', lambda x: x.corr(df.loc[x.index, 'monthly_rent_ars_num'])),
    corr_renov_precio=('renovation', lambda x: df.loc[x.index, 'renovation'].map({'no_renovation':0, 'cosmetic':1, 'euro':2, 'designer':3}).corr(df.loc[x.index, 'monthly_rent_ars_num']))
)
print(zone_stats.round(2))

print("\n=== EJEMPLO DEL STAKEHOLDER: CUARTITO EN EL CENTRO VS DEPTO GRANDE EN PERIFERIA ===")
# Cuartito en el centro: 1 room / area < 40m2 in Central (<5km)
centro_chico = df[(df['zone'] == 'Central (<5km)') & (df['total_area_m2'] <= 40)]
# Depto grande en periferia: 3-4 rooms / area >= 70m2 in Periferia (>15km)
perif_grande = df[(df['zone'] == 'Periferia (>15km)') & (df['total_area_m2'] >= 70)]

print(f"Centro chico (<= 40 m2, < 5km): n={len(centro_chico)}, Alquiler medio = ${centro_chico['monthly_rent_ars_num'].mean():,.0f} ARS, Mediana = ${centro_chico['monthly_rent_ars_num'].median():,.0f} ARS")
print(f"Periferia grande (>= 70 m2, > 15km): n={len(perif_grande)}, Alquiler medio = ${perif_grande['monthly_rent_ars_num'].mean():,.0f} ARS, Mediana = ${perif_grande['monthly_rent_ars_num'].median():,.0f} ARS")

# Check importance of features across zones using a Random Forest or Decision Tree feature importances
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

X_sample = df[['total_area_m2', 'rooms', 'building_year', 'renovation', 'building_type', 'furnished', 'to_center_km_real', 'metro_distance_min']].copy()
X_sample['renovation'] = OrdinalEncoder().fit_transform(X_sample[['renovation']])
X_sample['building_type'] = OrdinalEncoder().fit_transform(X_sample[['building_type']])
X_sample['furnished'] = X_sample['furnished'].astype(int)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_sample, df['monthly_rent_ars_num'])

importances = pd.Series(rf.feature_importances_, index=X_sample.columns).sort_values(ascending=False)
print("\n=== RANDOM FOREST GLOBAL FEATURE IMPORTANCES ===")
print(importances.round(4))
