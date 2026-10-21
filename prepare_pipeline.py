import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# 1. Carga de datos
df = pd.read_csv('rentals.txt', sep=';')

# 2. Conversión del precio bruto
df['rent_ars_nominal'] = df['monthly_rent_ars'].str.replace('$ ', '', regex=False).str.replace('.', '', regex=False).astype(float)
df['date_posted'] = pd.to_datetime(df['date_posted'], format='%d/%m/%Y')
df['year'] = df['date_posted'].dt.year

# 3. Homogeneización de Precios a Base 2026 (Deflactor Mediano)
# Justificación: Aislar el efecto macroeconómico para modelar el valor intrínseco del inmueble.
mediana_2026 = df.loc[df['year'] == 2026, 'rent_ars_nominal'].median()
deflactores = df.groupby('year')['rent_ars_nominal'].median().apply(lambda m: mediana_2026 / m).to_dict()
df['deflactor'] = df['year'].map(deflactores)
df['rent_ars_homogeneizado'] = df['rent_ars_nominal'] * df['deflactor']

# 4. Feature Engineering
df['total_area_m2'] = df['total_area'] / 10.0
df['to_center_km_real'] = df['to_center_km'] / 100.0
df['area_per_room'] = df['total_area_m2'] / df['rooms']
df['accessible_area'] = df['total_area_m2'] / (df['to_center_km_real'] + 1.0)
df['interaction_area_center'] = df['total_area_m2'] * df['to_center_km_real']
df['is_central'] = (df['to_center_km_real'] <= 5.0).astype(int)
df['building_age'] = 2026 - df['building_year']
df['renovation_code'] = df['renovation'].map({'no_renovation': 0, 'cosmetic': 1, 'euro': 2, 'designer': 3})
df['furnished'] = df['furnished'].astype(int)
df['pets_allowed'] = df['pets_allowed'].astype(int)

# 5. Partición Train / Test (80% Train, 20% Test) con fijación de semilla aleatoria (random_state=42)
# Muestreo estratificado por deciles del precio homogeneizado para garantizar idéntica distribución
df['price_decile'] = pd.qcut(df['rent_ars_homogeneizado'], q=10, labels=False)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['price_decile'])

# 6. Target Encoding Out-of-Fold para 'district' dentro de Train (¡Evitando Data Leakage al Test!)
# Smoothing bayesiano simple: (n * mean_group + m * global_mean) / (n + m) con m=10
global_mean = train_df['rent_ars_homogeneizado'].mean()
m = 10.0

kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_df = train_df.copy()
test_df = test_df.copy()
train_df['district_encoded'] = 0.0

for tr_idx, val_idx in kf.split(train_df):
    tr = train_df.iloc[tr_idx]
    stats = tr.groupby('district')['rent_ars_homogeneizado'].agg(['count', 'mean'])
    smoothed = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
    train_df.iloc[val_idx, train_df.columns.get_loc('district_encoded')] = train_df.iloc[val_idx]['district'].map(smoothed).fillna(global_mean)

# En test aplicamos el target encoding consolidado de todo train
full_train_stats = train_df.groupby('district')['rent_ars_homogeneizado'].agg(['count', 'mean'])
smoothed_full = (full_train_stats['count'] * full_train_stats['mean'] + m * global_mean) / (full_train_stats['count'] + m)
test_df['district_encoded'] = test_df['district'].map(smoothed_full).fillna(global_mean)

# One-hot encoding de building_type
ohe = OneHotEncoder(drop='first', sparse_output=False)
bt_train_ohe = pd.DataFrame(ohe.fit_transform(train_df[['building_type']]), index=train_df.index, columns=ohe.get_feature_names_out(['building_type']))
bt_test_ohe = pd.DataFrame(ohe.transform(test_df[['building_type']]), index=test_df.index, columns=ohe.get_feature_names_out(['building_type']))

train_final = pd.concat([train_df, bt_train_ohe], axis=1)
test_final = pd.concat([test_df, bt_test_ohe], axis=1)

features = [
    'total_area_m2', 'rooms', 'building_age', 'renovation_code',
    'furnished', 'pets_allowed', 'to_center_km_real', 'metro_distance_min',
    'area_per_room', 'accessible_area', 'interaction_area_center', 'is_central',
    'district_encoded'
] + list(bt_train_ohe.columns)

X_train = train_final[features]
y_train = train_final['rent_ars_homogeneizado']

X_test = test_final[features]
y_test = test_final['rent_ars_homogeneizado']

print("Dimensiones Finales:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print("Features utilizadas:", features)
