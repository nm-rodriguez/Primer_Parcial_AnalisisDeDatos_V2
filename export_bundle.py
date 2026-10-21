import pickle
import pandas as pd
import numpy as np

# Let's save the inference bundle so predecir_alquiler.py is fast and standalone
import run_models_mlflow as rmm

bundle = {
    'model': rmm.hgb,
    'feature_cols': rmm.feature_cols,
    'district_mapping': rmm.smoothed_full.to_dict(),
    'global_mean': rmm.global_mean,
    'ohe': rmm.ohe,
    'ohe_cols': list(rmm.bt_train_ohe.columns)
}

with open('model_bundle.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print("model_bundle.pkl saved successfully!")
