import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 10})
os.makedirs('eda_plots', exist_ok=True)

df = pd.read_csv('rentals.txt', sep=';')
df['monthly_rent_ars_num'] = df['monthly_rent_ars'].str.replace('$ ', '', regex=False).str.replace('.', '', regex=False).astype(float)
df['total_area_m2'] = df['total_area'] / 10.0
df['to_center_km_real'] = df['to_center_km'] / 100.0

# Plot 1: Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['monthly_rent_ars_num'] / 1e3, kde=True, ax=axes[0], color='#2b5c8f', bins=50)
axes[0].set_title('Distribución de Alquiler Mensual (Miles de ARS)')
axes[0].set_xlabel('Alquiler mensual (en miles de ARS)')
axes[0].set_ylabel('Frecuencia')

sns.boxplot(x=df['monthly_rent_ars_num'] / 1e3, ax=axes[1], color='#e77f67')
axes[1].set_title('Boxplot de Alquiler Mensual (Detección de Outliers)')
axes[1].set_xlabel('Alquiler mensual (en miles de ARS)')
plt.tight_layout()
plt.savefig('eda_plots/1_target_distribution.png', dpi=150)
plt.close()

# Plot 2: Correlation Heatmap
features_num = ['total_area_m2', 'rooms', 'floor', 'total_floors', 'building_year', 'metro_distance_min', 'to_center_km_real', 'deposit_months', 'monthly_rent_ars_num']
corr_labels = ['Área (m²)', 'Habitaciones', 'Piso', 'Pisos Totales', 'Año Constr.', 'Dist. Metro (min)', 'Dist. Centro (km)', 'Depósito (meses)', 'Precio (ARS)']
corr_mat = df[features_num].corr()
corr_mat.columns = corr_labels
corr_mat.index = corr_labels

plt.figure(figsize=(9, 7))
sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True, vmin=-1, vmax=1)
plt.title('Matriz de Correlación de Variables Numéricas')
plt.tight_layout()
plt.savefig('eda_plots/2_correlation_matrix.png', dpi=150)
plt.close()

# Plot 3: Price by Okrug
plt.figure(figsize=(12, 6))
okrug_order = df.groupby('okrug')['monthly_rent_ars_num'].median().sort_values(ascending=False).index
sns.boxplot(x='okrug', y=df['monthly_rent_ars_num'] / 1e3, data=df, order=okrug_order, palette='Spectral')
plt.title('Distribución de Precios de Alquiler por Okrug (Macrodistrito Administrativo)')
plt.xlabel('Okrug (ordenado por mediana decreciente)')
plt.ylabel('Alquiler mensual (en miles de ARS)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_plots/3_price_by_okrug.png', dpi=150)
plt.close()

# Plot 4: Scatter price vs area and price vs distance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(x='total_area_m2', y=df['monthly_rent_ars_num'] / 1e3, hue='okrug', data=df, alpha=0.5, ax=axes[0], legend=False)
axes[0].set_title('Relación: Superficie (m²) vs Precio Alquiler (miles ARS)')
axes[0].set_xlabel('Superficie Total (m²)')
axes[0].set_ylabel('Precio (miles ARS)')

sns.scatterplot(x='to_center_km_real', y=df['monthly_rent_ars_num'] / 1e3, hue='okrug', data=df, alpha=0.5, ax=axes[1], legend=False)
axes[1].set_title('Relación: Distancia al Centro (km) vs Precio Alquiler (miles ARS)')
axes[1].set_xlabel('Distancia al Centro (km)')
axes[1].set_ylabel('Precio (miles ARS)')
plt.tight_layout()
plt.savefig('eda_plots/4_price_vs_area_distance.png', dpi=150)
plt.close()

print("Plots successfully created in eda_plots/ folder.")
