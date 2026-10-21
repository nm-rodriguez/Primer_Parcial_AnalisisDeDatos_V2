import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import scipy.stats as stats

# Configuración visual
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})
os.makedirs('mlflow_artifacts', exist_ok=True)

# 1. Carga del Dataset
df = pd.read_csv('rentals.txt', sep=';')

# 2. Parseo de Fechas y Precios
df['rent_ars_nominal'] = df['monthly_rent_ars'].str.replace('$ ', '', regex=False).str.replace('.', '', regex=False).astype(float)
df['date_posted'] = pd.to_datetime(df['date_posted'], format='%d/%m/%Y')
df['year'] = df['date_posted'].dt.year

# 3. Homogeneización de Precios a Base 2026 (Deflactor Mediano)
mediana_2026 = df.loc[df['year'] == 2026, 'rent_ars_nominal'].median()
deflactores = df.groupby('year')['rent_ars_nominal'].median().apply(lambda m: mediana_2026 / m).to_dict()
df['deflactor'] = df['year'].map(deflactores)
df['rent_ars_homogeneizado'] = df['rent_ars_nominal'] * df['deflactor']

# 4. Feature Engineering Acordado
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

# 5. Muestreo Estratificado Train / Test (80/20) con random_state=42
df['price_decile'] = pd.qcut(df['rent_ars_homogeneizado'], q=10, labels=False)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['price_decile'])
train_df = train_df.copy()
test_df = test_df.copy()

# 6. Target Encoding Out-Of-Fold para 'district' (m=10 smoothing)
global_mean = train_df['rent_ars_homogeneizado'].mean()
m = 10.0
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_df['district_encoded'] = 0.0

for tr_idx, val_idx in kf.split(train_df):
    tr = train_df.iloc[tr_idx]
    stats_df = tr.groupby('district')['rent_ars_homogeneizado'].agg(['count', 'mean'])
    smoothed = (stats_df['count'] * stats_df['mean'] + m * global_mean) / (stats_df['count'] + m)
    train_df.iloc[val_idx, train_df.columns.get_loc('district_encoded')] = train_df.iloc[val_idx]['district'].map(smoothed).fillna(global_mean)

full_train_stats = train_df.groupby('district')['rent_ars_homogeneizado'].agg(['count', 'mean'])
smoothed_full = (full_train_stats['count'] * full_train_stats['mean'] + m * global_mean) / (full_train_stats['count'] + m)
test_df['district_encoded'] = test_df['district'].map(smoothed_full).fillna(global_mean)

# One-Hot Encoding para building_type
ohe = OneHotEncoder(drop='first', sparse_output=False)
bt_train_ohe = pd.DataFrame(ohe.fit_transform(train_df[['building_type']]), index=train_df.index, columns=ohe.get_feature_names_out(['building_type']))
bt_test_ohe = pd.DataFrame(ohe.transform(test_df[['building_type']]), index=test_df.index, columns=ohe.get_feature_names_out(['building_type']))

train_final = pd.concat([train_df, bt_train_ohe], axis=1)
test_final = pd.concat([test_df, bt_test_ohe], axis=1)

feature_cols = [
    'total_area_m2', 'rooms', 'building_age', 'renovation_code',
    'furnished', 'pets_allowed', 'to_center_km_real', 'metro_distance_min',
    'area_per_room', 'accessible_area', 'interaction_area_center', 'is_central',
    'district_encoded'
] + list(bt_train_ohe.columns)

X_train = train_final[feature_cols]
y_train = train_final['rent_ars_homogeneizado']

X_test = test_final[feature_cols]
y_test = test_final['rent_ars_homogeneizado']

# 7. Diagnóstico de Multicolinealidad (VIF y Tolerancia)
print("\n---> Calculando Factor de Inflación de la Varianza (VIF) y Tolerancia (TOL)...")
scaler_vif = StandardScaler()
X_train_scaled = scaler_vif.fit_transform(X_train)

vif_records = []
for idx, col in enumerate(feature_cols):
    vif_val = variance_inflation_factor(X_train_scaled, idx)
    tol_val = 1.0 / vif_val if vif_val > 0 else 0.0
    vif_records.append({
        "Variable": col,
        "VIF": round(float(vif_val), 3),
        "TOL": round(float(tol_val), 4),
        "Estado": "Alarma (>5)" if vif_val > 5 else "Aceptable"
    })
vif_table = pd.DataFrame(vif_records).sort_values(by="VIF", ascending=False)
vif_path = "mlflow_artifacts/vif_tolerance_table.csv"
vif_table.to_csv(vif_path, index=False)
global_max_vif = float(vif_table['VIF'].max())
global_mean_vif = float(vif_table['VIF'].mean())
global_cond_no = float(np.linalg.cond(X_train_scaled))
print(f"VIF Máximo: {global_max_vif:.2f} | VIF Promedio: {global_mean_vif:.2f} | Cond. No.: {global_cond_no:.2f}")

# 8. Reporte OLS Clásico Statsmodels (Referencia Didáctica Diapositivas 14, 22, 51)
print("\n---> Ajustando Modelo OLS Statsmodels de Referencia Teórica...")
X_train_sm = sm.add_constant(X_train_scaled)
ols_model_sm = sm.OLS(y_train, X_train_sm).fit()
ols_summary_path = "mlflow_artifacts/ols_statsmodels_summary.txt"
with open(ols_summary_path, "w", encoding="utf-8") as f:
    f.write(str(ols_model_sm.summary()))

ols_coefs_df = pd.DataFrame({
    "Variable": ["const"] + feature_cols,
    "coef": ols_model_sm.params.values,
    "std_err": ols_model_sm.bse.values,
    "t_value": ols_model_sm.tvalues.values,
    "p_value": ols_model_sm.pvalues.values,
    "ci_0.025": ols_model_sm.conf_int()[0].values,
    "ci_0.975": ols_model_sm.conf_int()[1].values
})
ols_coefs_path = "mlflow_artifacts/ols_coefficients.csv"
ols_coefs_df.to_csv(ols_coefs_path, index=False)

# Configuración del Experimento MLflow
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment_name = "prediccion_alquileres_rusia"
mlflow.set_experiment(experiment_name)

print(f"\nMLflow tracking activo: {experiment_name}")
print(f"Muestras Train: {len(X_train)}, Muestras Test: {len(X_test)}, Predictores: {len(feature_cols)}")

# Función de Evaluación y Registro Integral de Métricas de la Cátedra
def evaluate_and_log(run_name, model, is_log_target=False, params_dict=None):
    with mlflow.start_run(run_name=run_name):
        print(f"\n=======================================================")
        print(f"---> Ejecutando y Evaluando: {run_name}")
        print(f"=======================================================")
        
        # Log de Hiperparámetros
        if params_dict:
            mlflow.log_params(params_dict)
        mlflow.log_param("is_log_target", is_log_target)
        mlflow.log_param("num_features", len(feature_cols))
        
        # Ajuste del modelo
        if is_log_target:
            y_train_fit = np.log(y_train)
            model.fit(X_train, y_train_fit)
            y_train_pred = np.exp(model.predict(X_train))
            y_pred = np.exp(model.predict(X_test))
        else:
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_pred = model.predict(X_test)
            
        # 1. Métricas de Rendimiento y Bondad de Ajuste
        n_test = len(y_test)
        n_train = len(y_train)
        k_features = len(feature_cols)
        
        r2_test = float(r2_score(y_test, y_pred))
        adj_r2_test = float(1.0 - (1.0 - r2_test) * (n_test - 1) / (n_test - k_features - 1))
        r2_train = float(r2_score(y_train, y_train_pred))
        adj_r2_train = float(1.0 - (1.0 - r2_train) * (n_train - 1) / (n_train - k_features - 1))
        
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100.0)
        rss = float(np.sum((y_test - y_pred) ** 2))
        tss = float(np.sum((y_test - np.mean(y_test)) ** 2))
        
        # F-statistic global
        if rss > 0 and (n_test - k_features - 1) > 0:
            f_stat = float(((tss - rss) / k_features) / (rss / (n_test - k_features - 1)))
            f_pvalue = float(stats.f.sf(f_stat, k_features, n_test - k_features - 1))
        else:
            f_stat, f_pvalue = 0.0, 1.0
            
        # Criterios de Información (AIC / BIC / Log-Likelihood)
        sigma2 = rss / n_test
        llf = float(-0.5 * n_test * (np.log(2 * np.pi) + np.log(sigma2) + 1.0))
        aic = float(2 * (k_features + 1) - 2 * llf)
        bic = float(np.log(n_test) * (k_features + 1) - 2 * llf)
        
        # 2. Diagnóstico de Residuos
        residuals = y_test - y_pred
        res_mean = float(np.mean(residuals))
        res_median = float(np.median(residuals))
        res_std = float(np.std(residuals))
        dw = float(durbin_watson(residuals))
        sk = float(stats.skew(residuals))
        kt_pearson = float(stats.kurtosis(residuals, fisher=False)) # Normal = 3
        kt_fisher = float(stats.kurtosis(residuals, fisher=True))   # Normal = 0
        
        jb_stat, jb_pvalue, _, _ = jarque_bera(residuals)
        omni_stat, omni_pvalue = omni_normtest(residuals)
        
        # Log de todas las métricas en MLflow
        # Rendimiento predictivo
        mlflow.log_metric("test_r2", r2_test)
        mlflow.log_metric("test_adj_r2", adj_r2_test)
        mlflow.log_metric("train_r2", r2_train)
        mlflow.log_metric("train_adj_r2", adj_r2_train)
        mlflow.log_metric("test_mae_ars", mae)
        mlflow.log_metric("test_rmse_ars", rmse)
        mlflow.log_metric("test_mse_ars", mse)
        mlflow.log_metric("test_mape_percent", mape)
        mlflow.log_metric("test_rss_ars", rss)
        
        # Significancia y Complejidad
        mlflow.log_metric("test_f_statistic", f_stat)
        mlflow.log_metric("test_prob_f_statistic", f_pvalue)
        mlflow.log_metric("aic", aic)
        mlflow.log_metric("bic", bic)
        mlflow.log_metric("log_likelihood", llf)
        
        # Residuos y Supuestos Teóricos
        mlflow.log_metric("durbin_watson", dw)
        mlflow.log_metric("residual_mean", res_mean)
        mlflow.log_metric("residual_median", res_median)
        mlflow.log_metric("skewness", sk)
        mlflow.log_metric("kurtosis", kt_pearson)
        mlflow.log_metric("kurtosis_fisher", kt_fisher)
        mlflow.log_metric("jarque_bera_stat", float(jb_stat))
        mlflow.log_metric("jarque_bera_pvalue", float(jb_pvalue))
        mlflow.log_metric("omnibus_stat", float(omni_stat))
        mlflow.log_metric("omnibus_pvalue", float(omni_pvalue))
        
        # Multicolinealidad
        mlflow.log_metric("max_vif", global_max_vif)
        mlflow.log_metric("mean_vif", global_mean_vif)
        mlflow.log_metric("condition_number", global_cond_no)
        
        print(f"Resultados Principales:")
        print(f"  R2: {r2_test:.4f} | R2 Ajustado: {adj_r2_test:.4f}")
        print(f"  MAE: ${mae:,.0f} ARS | RMSE: ${rmse:,.0f} ARS | MAPE: {mape:.2f}%")
        print(f"  AIC: {aic:,.1f} | BIC: {bic:,.1f}")
        print(f"  Durbin-Watson: {dw:.3f} (Esperado 1.5 - 2.5)")
        print(f"  Skewness: {sk:.3f} | Kurtosis Fisher: {kt_fisher:.3f}")
        print(f"  Jarque-Bera p-value: {jb_pvalue:.4e} | Omnibus p-value: {omni_pvalue:.4e}")
        
        # 3. Artefactos Gráficos de Diagnóstico (Panel 2x2 correspondiente a las diapositivas)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Cuadrante 1: Real vs Predicho
        axes[0, 0].scatter(y_test / 1e3, y_pred / 1e3, alpha=0.25, color='#2980b9', s=16)
        min_v = min(y_test.min(), y_pred.min()) / 1e3
        max_v = max(y_test.max(), y_pred.max()) / 1e3
        axes[0, 0].plot([min_v, max_v], [min_v, max_v], color='#e74c3c', linestyle='--', linewidth=2, label='Ideal 45°')
        axes[0, 0].set_title(f'1. Real vs Predicho (R² = {r2_test:.4f})', fontweight='bold')
        axes[0, 0].set_xlabel('Alquiler Real (Miles ARS)')
        axes[0, 0].set_ylabel('Alquiler Predicho (Miles ARS)')
        axes[0, 0].legend()
        
        # Cuadrante 2: Histograma de Residuos (Diapositiva 30)
        sns.histplot(residuals / 1e3, kde=True, ax=axes[0, 1], color='#8e44ad', bins=40, stat="density")
        # Curva normal teórica de comparación
        mu_res, std_res = res_mean / 1e3, res_std / 1e3
        x_norm = np.linspace(residuals.min() / 1e3, residuals.max() / 1e3, 200)
        p_norm = stats.norm.pdf(x_norm, mu_res, std_res)
        axes[0, 1].plot(x_norm, p_norm, 'r--', linewidth=2, label='Normal Teórica')
        axes[0, 1].axvline(0, color='black', linestyle=':', label='Cero')
        axes[0, 1].set_title(f'2. Histograma de Residuos (Skew={sk:.2f}, Curt={kt_fisher:.2f})', fontweight='bold')
        axes[0, 1].set_xlabel('Residuo e = y - ŷ (Miles ARS)')
        axes[0, 1].legend()
        
        # Cuadrante 3: Residuales vs Valores Ajustados (Diapositivas 32-37)
        axes[1, 0].scatter(y_pred / 1e3, residuals / 1e3, alpha=0.25, color='#16a085', s=16)
        axes[1, 0].axhline(0, color='#e74c3c', linestyle='--', linewidth=2)
        # Tendencia móvil
        sorted_idx = np.argsort(y_pred)
        pred_sorted = (y_pred.iloc[sorted_idx] if hasattr(y_pred, 'iloc') else y_pred[sorted_idx]) / 1e3
        res_sorted = (residuals.iloc[sorted_idx] if hasattr(residuals, 'iloc') else residuals[sorted_idx]) / 1e3
        smooth_res = pd.Series(res_sorted).rolling(window=150, min_periods=30, center=True).mean()
        axes[1, 0].plot(pred_sorted, smooth_res, color='blue', linewidth=2, label='Tendencia / Sesgo')
        axes[1, 0].set_title('3. Residuales vs Valores Ajustados (Homocedasticidad)', fontweight='bold')
        axes[1, 0].set_xlabel('Valores Ajustados ŷ (Miles ARS)')
        axes[1, 0].set_ylabel('Residuos e (Miles ARS)')
        axes[1, 0].legend()
        
        # Cuadrante 4: Normal Q-Q Plot (Diapositivas 39-42)
        sm.qqplot(residuals, stats.t if False else stats.norm, fit=True, line='45', ax=axes[1, 1],
                  markerfacecolor='#2c3e50', markeredgecolor='none', alpha=0.3, markersize=4)
        axes[1, 1].get_lines()[1].set_color('#e74c3c')
        axes[1, 1].get_lines()[1].set_linewidth(2)
        axes[1, 1].set_title(f'4. Q-Q Plot Normal (JB p-val: {jb_pvalue:.2e})', fontweight='bold')
        axes[1, 1].set_xlabel('Cuantiles Teóricos')
        axes[1, 1].set_ylabel('Cuantiles Muestrales de Residuos')
        
        plt.tight_layout()
        diag_path = f"mlflow_artifacts/{run_name}_diagnostico_completo.png"
        plt.savefig(diag_path, dpi=120)
        plt.close()
        mlflow.log_artifact(diag_path)
        
        # Scale-Location Plot (Diapositiva 43)
        std_residuals = residuals / res_std
        sqrt_abs_std_res = np.sqrt(np.abs(std_residuals))
        fig_sl, ax_sl = plt.subplots(figsize=(9, 5))
        ax_sl.scatter(y_pred / 1e3, sqrt_abs_std_res, alpha=0.3, color='#34495e', s=16)
        smooth_sl = pd.Series(sqrt_abs_std_res.iloc[sorted_idx] if hasattr(sqrt_abs_std_res, 'iloc') else sqrt_abs_std_res[sorted_idx]).rolling(window=150, min_periods=30, center=True).mean()
        ax_sl.plot(pred_sorted, smooth_sl, color='#e74c3c', linewidth=2.5, label='Línea de Tendencia')
        ax_sl.set_title(f'{run_name}: Scale-Location Plot (Homocedasticidad)', fontweight='bold')
        ax_sl.set_xlabel('Valores Ajustados ŷ (Miles ARS)')
        ax_sl.set_ylabel('√|Residuos Estandarizados|')
        ax_sl.legend()
        plt.tight_layout()
        sl_path = f"mlflow_artifacts/{run_name}_scale_location.png"
        plt.savefig(sl_path, dpi=120)
        plt.close()
        mlflow.log_artifact(sl_path)
        
        # Feature Importance / Coeficientes
        imp_path = f"mlflow_artifacts/{run_name}_feature_importance.png"
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
            plt.figure(figsize=(10, 6))
            importances.tail(12).plot(kind='barh', color='#2ecc71')
            plt.title(f'{run_name}: Top 12 Variables más Importantes (MDI)', fontweight='bold')
            plt.xlabel('Importancia Relativa')
            plt.tight_layout()
            plt.savefig(imp_path, dpi=120)
            plt.close()
            mlflow.log_artifact(imp_path)
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('regressor', None), 'coef_'):
            coefs = pd.Series(model.named_steps['regressor'].coef_, index=feature_cols).sort_values()
            plt.figure(figsize=(10, 6))
            coefs.plot(kind='barh', color='#3498db')
            plt.title(f'{run_name}: Coeficientes Estandarizados', fontweight='bold')
            plt.xlabel('Magnitud del Coeficiente')
            plt.tight_layout()
            plt.savefig(imp_path, dpi=120)
            plt.close()
            mlflow.log_artifact(imp_path)
            
        # Log de Artefactos de Diagnóstico Globales
        mlflow.log_artifact(vif_path)
        mlflow.log_artifact(ols_summary_path)
        mlflow.log_artifact(ols_coefs_path)
        
        # Serializar modelo en MLflow
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        
        return {
            "Modelo": run_name,
            "R2": round(r2_test, 4),
            "Adj R2": round(adj_r2_test, 4),
            "MAE (ARS)": f"${mae:,.0f}",
            "RMSE (ARS)": f"${rmse:,.0f}",
            "MAPE (%)": f"{mape:.2f}%",
            "AIC": round(aic, 1),
            "BIC": round(bic, 1),
            "Durbin-Watson": round(dw, 3),
            "Skewness": round(sk, 3),
            "Kurtosis Fisher": round(kt_fisher, 3),
            "Prob(JB)": f"{jb_pvalue:.2e}"
        }

# --- EJECUCIÓN DE LAS ITERACIONES EXPERIMENTALES ---
summary_records = []

# Iteración 0: Baseline Regresión Lineal OLS Clásica (MCO sin regularización)
ols_baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
res0 = evaluate_and_log(
    run_name="Iteracion_0_OLS_Lineal",
    model=ols_baseline,
    is_log_target=False,
    params_dict={"model_type": "LinearRegression_OLS", "regularization": "None", "scaler": "StandardScaler"}
)
summary_records.append(res0)

# Iteración 1: Regresión Lineal Ridge con regularización L2 sobre Escala Lineal
ridge_baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=100.0, random_state=42))
])
res1 = evaluate_and_log(
    run_name="Iteracion_1_Ridge_Lineal",
    model=ridge_baseline,
    is_log_target=False,
    params_dict={"model_type": "Ridge", "alpha": 100.0, "scaler": "StandardScaler"}
)
summary_records.append(res1)

# Iteración 2: Regresión Lineal Ridge con Transformación Logarítmica Log(y)
ridge_log = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=100.0, random_state=42))
])
res2 = evaluate_and_log(
    run_name="Iteracion_2_Ridge_LogTarget",
    model=ridge_log,
    is_log_target=True,
    params_dict={"model_type": "Ridge_LogTarget", "alpha": 100.0, "scaler": "StandardScaler"}
)
summary_records.append(res2)

# Iteración 3: Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=16,
    min_samples_split=6,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
res3 = evaluate_and_log(
    run_name="Iteracion_3_Random_Forest",
    model=rf,
    is_log_target=False,
    params_dict={
        "model_type": "RandomForestRegressor",
        "n_estimators": 100,
        "max_depth": 16,
        "min_samples_split": 6,
        "min_samples_leaf": 3,
        "max_features": "sqrt"
    }
)
summary_records.append(res3)

# Iteración 4: HistGradientBoosting Regressor (Modelo Campeón)
hgb = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.08,
    max_depth=12,
    min_samples_leaf=20,
    l2_regularization=1.5,
    random_state=42
)
res4 = evaluate_and_log(
    run_name="Iteracion_4_Gradient_Boosting",
    model=hgb,
    is_log_target=False,
    params_dict={
        "model_type": "HistGradientBoostingRegressor",
        "max_iter": 200,
        "learning_rate": 0.08,
        "max_depth": 12,
        "min_samples_leaf": 20,
        "l2_regularization": 1.5
    }
)
summary_records.append(res4)

# Resumen Final Comparativo
summary_df = pd.DataFrame(summary_records)
print("\n" + "="*80)
print("TABLA COMPARATIVA FINAL DE TODAS LAS MÉRICAS TRACKEADAS EN MLFLOW")
print("="*80)
print(summary_df.to_string(index=False))

summary_df.to_csv('mlflow_artifacts/summary_table.csv', index=False)
print("\nProceso finalizado exitosamente. Todos los artefactos y métricas han sido registrados en MLflow.")
