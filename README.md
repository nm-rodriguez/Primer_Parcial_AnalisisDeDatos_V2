# Sistema Predictivo de Tasación de Alquileres Inmobiliarios (Moscú)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![MLflow Tracking](https://img.shields.io/badge/Tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Pipeline integral de Machine Learning para la estimación y tasación precisa de precios de alquiler residencial en la ciudad de Moscú, Rusia. El proyecto aborda la decodificación de datos físicos, resolución de multicolinealidad, feature engineering geoespacial, muestreo estratificado sin fuga de datos (*Data Leakage*) y experimentación iterativa registrada en **MLflow**.

---

## 🏆 Modelo Campeón y Rendimiento

El modelo seleccionado para producción es un **HistGradientBoosting Regressor** optimizado con regularización $L_2$ y binning por histogramas, evaluado sobre un conjunto de prueba ciego de 4.000 propiedades (20% del total):

| Métrica | Valor Obtenido | Interpretación Técnica |
| :--- | :---: | :--- |
| **$R^2$ Score** | **$0.9527$** | Explica el $95.27\%$ de la variabilidad del mercado inmobiliario. |
| **MAE** | **$\$84.428$ ARS** | Error absoluto medio por propiedad (reducción de $\$52.277$ vs. baseline). |
| **RMSE** | **$\$122.982$ ARS** | Penalización acotada sobre desviaciones extremas. |
| **MAPE** | **$8.56\%$** | Desviación porcentual promedio inferior al $9\%$ (grado de excelencia comercial). |
| **Durbin-Watson** | **$1.938$** | Ausencia total de autocorrelación en los residuos ($\approx 2.0$). |

---

## 📁 Estructura del Repositorio

```text
├── rentals.txt                  # Dataset crudo (20.000 observaciones, 23 variables)
├── prepare_pipeline.py          # Script de preprocesamiento, deflactación y partición train/test
├── run_models_mlflow.py         # Pipeline de experimentación y registro de corridas en MLflow
├── export_bundle.py             # Generación y validación del artefacto productivo (bundle)
├── build_inference_assets.py    # Pipeline de construcción de transformadores para inferencia
├── predecir_alquiler.py         # Tasador interactivo por consola (CLI de producción)
├── model_bundle.pkl             # Artefacto binario serializado (modelo + transformadores OOF)
├── iniciar_mlflow.ps1           # Script PowerShell para iniciar el Dashboard de MLflow
├── iniciar_mlflow.bat           # Script Batch para iniciar el Dashboard de MLflow en Windows
├── field_analysis.py            # Análisis de correlaciones, ANOVA distritos vs. okrugs
├── stakeholder_analysis.py      # Validación empírica de hipótesis (centro vs. periferia)
├── deep_eda.py                  # Auditoría de calidad de datos, nulos, duplicados y outliers
├── mlflow_artifacts/            # Diagnósticos residuales, curvas Q-Q e importancias exportadas
├── mlflow.db                    # Base de datos relacional SQLite de MLflow
├── .gitignore                   # Exclusión de temporales, reportes y material académico
└── README.md                    # Documentación técnica principal del proyecto
```

---

## ⚙️ Requisitos e Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DIRECTORIO>
   ```

2. **Crear y activar un entorno virtual (recomendado):**
   ```bash
   python -m venv .venv
   # En Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # En Linux / macOS:
   source .venv/bin/activate
   ```

3. **Instalar dependencias necesarias:**
   ```bash
   pip install numpy pandas scikit-learn mlflow statsmodels matplotlib seaborn cloudpickle
   ```

---

## 🚀 Uso Rápido: Tasador Inmobiliario por Consola (`predecir_alquiler.py`)

El script [`predecir_alquiler.py`](predecir_alquiler.py) es autónomo y consume directamente el artefacto [`model_bundle.pkl`](model_bundle.pkl).

### Opción 1: Ejemplo Demostrativo Instantáneo
```bash
python predecir_alquiler.py --ejemplo
```

### Opción 2: Modo Interactivo Guiado (Paso a Paso)
Si se ejecuta sin parámetros, el sistema solicita amablemente los datos por consola:
```bash
python predecir_alquiler.py
```

### Opción 3: Pasando Parámetros por Argumentos CLI
```bash
python predecir_alquiler.py \
  --district "Arbat" \
  --area 50 \
  --rooms 2 \
  --year 2015 \
  --building_type "monolith" \
  --renovation "designer" \
  --to_center_km 2.1 \
  --metro_dist 5 \
  --furnished
```

**Ejemplo de Salida Generada:**
```text
=================================================================
      [*] RESULTADO DE LA TASACION INMOBILIARIA PREDICTIVA
=================================================================
Caracteristicas del Inmueble Ingresado:
  - Distrito: Arbat
  - Superficie m2: 50.0
  - Habitaciones: 2
  - Antiguedad edificio: 11 años (Año 2015)
  - Tipo edificio: monolith
  - Remodelacion: designer
  - Distancia centro: 2.10 km
  - Distancia metro: 5 min
  - Amoblado: Sí
  - Mascotas: No
-----------------------------------------------------------------
  [ARS] ALQUILER ESTIMADO:      $ 3,444,855 ARS / mes
  [RUB] EQUIVALENTE EN RUBLOS:  191,381 RUB / mes  (tasa: 18 ARS/RUB)
  [m2]  PRECIO POR METRO:       $ 68,897 ARS / m2
-----------------------------------------------------------------
  Rango de Tasacion Esperado (+/- 1 MAE de $ 84,428):
   $ 3,360,427  <--->  $ 3,529,283 ARS / mes
=================================================================
```

---

## 🔬 Metodología y Pipeline de Modelado

### 1. Auditoría y Corrección de Escalas
- **`total_area`**: Almacenada originalmente en décimas de metro cuadrado ($/10$). Se reescaló a metros cuadrados reales ($m^2$).
- **`to_center_km`**: Almacenada originalmente en centésimas de kilómetro ($/100$). Se reescaló a kilómetros reales ($km$).
- **Prevención de Data Leakage**: Se eliminaron de raíz `rent_per_sqm` (fuga directa del target) y `monthly_rent_rub` (redundancia determinística de escala $18\times$).

### 2. Homogeneización Inflacionaria a Precios Constantes (Base 2026)
Para evitar que el modelo extrapole una tendencia inflacionaria artificial en producción, se descartó el año de publicación como variable libre y se deflactó la serie mediante el **índice mediano anual** a moneda de poder adquisitivo constante de 2026:
$$\text{Factor Deflactor}(t) = \frac{\text{Mediana}(\text{Precio}_{2026})}{\text{Mediana}(\text{Precio}_t)}$$

### 3. Muestreo Estratificado y Target Encoding Out-Of-Fold (OOF)
- Partición: **80% Entrenamiento (16.000)** y **20% Test Ciego (4.000)** estratificado por deciles del precio homogeneizado.
- Para los 129 distritos urbanos se aplicó un esquema **Out-of-Fold con 5 pliegues cruzados y suavizado bayesiano ($m=10$)** ajustado estrictamente sobre el conjunto de entrenamiento, evitando toda fuga de información hacia el test ciego.

### 4. Variables de Interacción Creadas
- `accessible_area`: $\frac{\text{total\_area\_m2}}{\text{to\_center\_km} + 1.0}$ (superficie ponderada por accesibilidad urbana).
- `interaction_area_center`: $\text{total\_area\_m2} \times \text{to\_center\_km}$ (interacción física continua).
- `area_per_room`: $\frac{\text{total\_area\_m2}}{\text{rooms}}$ (amplitud media por ambiente).
- `is_central`: Indicador binario para propiedades dentro del primer anillo de 5 km.
- `building_age`: Antigüedad física ($2026 - \text{building\_year}$).

---

## 📊 Ciclo de Iteraciones Experimentales en MLflow

Se ejecutaron 4 iteraciones progresivas registradas con parámetros, métricas y diagnósticos residuales en MLflow:

```text
Iteración 1 (Baseline)           Iteración 2 (Log-Target)           Iteración 3 (Bagging)           Iteración 4 (Boosting)
  [Ridge Lineal]       ──>        [Ridge Log(y)]          ──>       [Random Forest]       ──>     [HistGradientBoosting]
  R²=0.8689 | MAE=$136k            R²=0.8940 | MAE=$116k             R²=0.9133 | MAE=$109k           R²=0.9527 | MAE=$84k
 (Línea base aditiva)            (Homocedasticidad/Multiplicativa)  (No linealidad/Interacciones)     (Minimización secuencial 🏆)
```

| Iteración | Algoritmo | Target | $R^2$ | Adj $R^2$ | MAE (ARS) | RMSE (ARS) | MAPE (%) | Durbin-Watson |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iteración 1** | Ridge Lineal ($\alpha=100$) | Lineal | 0.8689 | 0.8683 | \$136.705 | \$204.727 | 15.75% | 2.011 |
| **Iteración 2** | Ridge Log-Target ($\alpha=100$) | $\log(y)$ | 0.8940 | 0.8936 | \$116.168 | \$184.025 | 11.18% | 1.955 |
| **Iteración 3** | Random Forest ($n=100$) | Lineal | 0.9133 | 0.9129 | \$109.709 | \$166.443 | 11.30% | 1.978 |
| **Iteración 4** | **HistGradientBoosting** | Lineal | **0.9527** | **0.9525** | **\$84.428** | **\$122.982** | **8.56%** | **1.938** |

---

## 📈 Visualización del Tracking con MLflow

Para inspeccionar las corridas, métricas comparativas y artefactos gráficos generados (`diagnostico_completo.png`, `feature_importance.png`, `scale_location.png`):

```bash
# Iniciar servidor UI de MLflow conectado a la base SQLite
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Luego ingresar desde el navegador web a:
👉 `http://localhost:5000`

---

## 🛠️ Reentrenamiento del Pipeline Completo

Para reejecutar todo el flujo experimental y regenerar el bundle productivo:

```bash
# 1. Ejecutar el pipeline de modelos y registrar en MLflow
python run_models_mlflow.py

# 2. Reconstruir y validar el artefacto de inferencia
python build_inference_assets.py
python export_bundle.py
```

---

## 📄 Licencia y Uso
Este repositorio ha sido desarrollado con fines académicos y profesionales para la asignatura **Análisis de Datos** (Ingeniería en Sistemas de Información). El código y los modelos están disponibles bajo la licencia MIT.
