import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

# 1. Cargar y preparar datos
df = pd.read_csv('rentals.txt', sep=';')
df['rent_ars_nominal'] = df['monthly_rent_ars'].str.replace('$ ', '', regex=False).str.replace('.', '', regex=False).astype(float)
df['date_posted'] = pd.to_datetime(df['date_posted'], format='%d/%m/%Y')
df['year'] = df['date_posted'].dt.year

mediana_2026 = df.loc[df['year'] == 2026, 'rent_ars_nominal'].median()
deflactores = df.groupby('year')['rent_ars_nominal'].median().apply(lambda m: mediana_2026 / m).to_dict()
df['deflactor'] = df['year'].map(deflactores)
df['rent_ars_homogeneizado'] = df['rent_ars_nominal'] * df['deflactor']

# Features base
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

df['price_decile'] = pd.qcut(df['rent_ars_homogeneizado'], q=10, labels=False)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['price_decile'])
train_df = train_df.copy()

# Target Encoding para distritos
global_mean = float(train_df['rent_ars_homogeneizado'].mean())
m = 10.0
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_df['district_encoded'] = 0.0

for tr_idx, val_idx in kf.split(train_df):
    tr = train_df.iloc[tr_idx]
    stats = tr.groupby('district')['rent_ars_homogeneizado'].agg(['count', 'mean'])
    smoothed = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
    train_df.iloc[val_idx, train_df.columns.get_loc('district_encoded')] = train_df.iloc[val_idx]['district'].map(smoothed).fillna(global_mean)

full_train_stats = train_df.groupby('district')['rent_ars_homogeneizado'].agg(['count', 'mean'])
smoothed_full = ((full_train_stats['count'] * full_train_stats['mean'] + m * global_mean) / (full_train_stats['count'] + m)).to_dict()

# One Hot Encoder para building_type
ohe = OneHotEncoder(drop='first', sparse_output=False)
bt_train_ohe = pd.DataFrame(ohe.fit_transform(train_df[['building_type']]), index=train_df.index, columns=ohe.get_feature_names_out(['building_type']))
train_final = pd.concat([train_df, bt_train_ohe], axis=1)

feature_cols = [
    'total_area_m2', 'rooms', 'building_age', 'renovation_code',
    'furnished', 'pets_allowed', 'to_center_km_real', 'metro_distance_min',
    'area_per_room', 'accessible_area', 'interaction_area_center', 'is_central',
    'district_encoded'
] + list(bt_train_ohe.columns)

X_train = train_final[feature_cols]
y_train = train_final['rent_ars_homogeneizado']

# Entrenar modelo campeón
hgb = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.08,
    max_depth=12,
    min_samples_leaf=20,
    l2_regularization=1.5,
    random_state=42
)
hgb.fit(X_train, y_train)

# Guardar paquete de inferencia
bundle = {
    'model': hgb,
    'feature_cols': feature_cols,
    'district_mapping': smoothed_full,
    'global_mean': global_mean,
    'ohe': ohe,
    'ohe_cols': list(bt_train_ohe.columns),
    'known_districts': sorted(list(df['district'].unique()))
}

with open('model_bundle.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print("Inference bundle successfully created: model_bundle.pkl")
