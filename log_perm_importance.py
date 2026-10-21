import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import cloudpickle

# Load test data and model
mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('prediccion_alquileres_rusia')
runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"])
run_4 = [r for r in runs if r.info.run_name == 'Iteracion_4_Gradient_Boosting'][0]

print(f"Run 4 ID: {run_4.info.run_id}")

# Compute permutation importance
import run_models_mlflow as rmm
X_test = rmm.X_test
y_test = rmm.y_test
hgb = rmm.hgb
feature_cols = rmm.feature_cols

perm_imp = permutation_importance(hgb, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
sorted_idx = perm_imp.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.barh(np.array(feature_cols)[sorted_idx][-12:], perm_imp.importances_mean[sorted_idx][-12:], color='#e67e22')
plt.title('Iteracion_4_Gradient_Boosting: Top 12 Permutation Feature Importance')
plt.xlabel('Disminución en R2 al permutar la variable')
plt.tight_layout()
imp4_path = 'mlflow_artifacts/Iteracion_4_Gradient_Boosting_feature_importance.png'
plt.savefig(imp4_path, dpi=120)
plt.close()

# Log to Run 4 in MLflow
client.log_artifact(run_4.info.run_id, imp4_path)
print("Permutation importance plot generated and logged to MLflow successfully!")
