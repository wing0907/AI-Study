import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import optuna
from xgboost import XGBRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib

# -----------------------------
# Configuration
# -----------------------------
N_LAGS = 5
TEST_DIR = r'C:\Study25\_data\\test\\'
DATA_PATH_H = os.path.join(TEST_DIR, '한화에어로스페이스 250711.csv')
DATA_PATH_S = os.path.join(TEST_DIR, '삼성전자 250711.csv')
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# 1. Load and preprocess data
# -----------------------------
df_h = pd.read_csv(DATA_PATH_H, encoding='cp949')
df_s = pd.read_csv(DATA_PATH_S, encoding='cp949')
df_h['일자'] = pd.to_datetime(df_h['일자'], format='%Y/%m/%d')
df_s['일자'] = pd.to_datetime(df_s['일자'], format='%Y/%m/%d')
df_h.sort_values('일자', inplace=True)
df_s.sort_values('일자', inplace=True)
df_h = df_h[['일자', '시가', '종가']].rename(columns={'시가':'open_h_원', '종가':'close_h_원'})
df_s = df_s[['일자', '종가']].rename(columns={'종가':'close_s_원'})
_df = pd.merge(df_h, df_s, on='일자', how='inner')
for lag in range(1, N_LAGS+1):
    _df[f'close_h_lag_{lag}_원'] = _df['close_h_원'].shift(lag)
_df['target_open_h_next_원'] = _df['open_h_원'].shift(-1)
_df.dropna(inplace=True)
feature_cols = [c for c in _df.columns if c.startswith('close_h_lag_')]
# Lag features to numeric (remove commas and convert to float)
_df[feature_cols] = _df[feature_cols].replace({',': ''}, regex=True).astype(float)
# Target to numeric (remove commas)
_df['target_open_h_next_원'] = _df['target_open_h_next_원'].replace({',': ''}, regex=True).astype(float)
X = _df[feature_cols][feature_cols]
y = _df['target_open_h_next_원']
date_cutoff = pd.to_datetime('2025-07-10')
mask_train = _df['일자'] <= date_cutoff
X_train, y_train = X[mask_train], y[mask_train]

# -----------------------------
# 2. TimeSeries CV
# -----------------------------
tscv = TimeSeriesSplit(n_splits=3)

# -----------------------------
# 3. Define Optuna objectives with XGBoost CV-based early stopping
# -----------------------------
def objective_factory(model_key, param_suggestions):
    def objective(trial):
        params = {name: spec(trial) for name, spec in param_suggestions.items()}
        params['random_state'] = 42
        if model_key == 'xgb':
            # Use xgb.cv to find optimal n_estimators
            dtrain = xgb.DMatrix(X_train, label=y_train)
            cv_results = xgb.cv(
                params, dtrain,
                num_boost_round=params.pop('n_estimators'),
                nfold=3,
                early_stopping_rounds=20,
                metrics='mae',
                seed=42,
                verbose_eval=False
            )
            best_rounds = len(cv_results)
            params['n_estimators'] = best_rounds
            model = XGBRegressor(**params)
        else:
            model = LGBMRegressor(**params) if model_key=='lgbm' else CatBoostRegressor(**params, verbose=False)
        # Cross-validate manually
        maes = []
        for tr, val in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[tr], X_train.iloc[val]
            y_tr, y_val = y_train.iloc[tr], y_train.iloc[val]
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            maes.append(mean_absolute_error(y_val, pred))
        return np.mean(maes)
    return objective

# Parameter specs
xgb_params = {
    'n_estimators': lambda t: t.suggest_int('n_estimators', 50, 300),
    'max_depth': lambda t: t.suggest_int('max_depth', 3, 12),
    'learning_rate': lambda t: t.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'subsample': lambda t: t.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': lambda t: t.suggest_float('colsample_bytree', 0.6, 1.0)
}
lgb_params = xgb_params.copy()
cat_params = {
    'iterations': lambda t: t.suggest_int('iterations', 100, 500),
    'depth': lambda t: t.suggest_int('depth', 3, 12),
    'learning_rate': lambda t: t.suggest_float('learning_rate', 0.01, 0.3, log=True)
}

# Run studies
studies = {}
studies['xgb'] = optuna.create_study(direction='minimize')
studies['xgb'].optimize(objective_factory('xgb', xgb_params), n_trials=30)
studies['lgbm'] = optuna.create_study(direction='minimize')
studies['lgbm'].optimize(objective_factory('lgbm', lgb_params), n_trials=30)
studies['cat'] = optuna.create_study(direction='minimize')
studies['cat'].optimize(objective_factory('cat', cat_params), n_trials=30)

# -----------------------------
# 4. Train final models and save
# -----------------------------
models = {}
for key, study in studies.items():
    best_params = study.best_params
    best_params['random_state'] = 42
    if key == 'xgb':
        # Re-run CV for full training to get final n_estimators
        dtrain = xgb.DMatrix(X_train, label=y_train)
        cv_results = xgb.cv(
            best_params, dtrain,
            num_boost_round=best_params.pop('n_estimators'),
            nfold=3,
            early_stopping_rounds=20,
            metrics='mae',
            seed=42,
            verbose_eval=False
        )
        best_params['n_estimators'] = len(cv_results)
        model = XGBRegressor(**best_params)
    elif key == 'lgbm':
        model = LGBMRegressor(**best_params)
    else:
        model = CatBoostRegressor(**best_params, verbose=False)
    model.fit(X_train, y_train)
    models[key] = model
    joblib.dump(model, os.path.join(MODEL_DIR, f'{key}_model.pkl'))

# -----------------------------
# 5. Ensemble weights (MAE 기반)
# -----------------------------
preds_train = {k: m.predict(X_train) for k, m in models.items()}
maes = {k: mean_absolute_error(y_train, p) for k, p in preds_train.items()}
inv = np.array([1/m for m in maes.values()])
weights_alg = inv / inv.sum()
ensemble_weights = dict(zip(models.keys(), weights_alg.tolist()))
with open(os.path.join(MODEL_DIR, 'ensemble_weights.json'), 'w') as f:
    json.dump(ensemble_weights, f)

# -----------------------------
# 6. Predict next open (정수 KRW)
# -----------------------------
X_pred = X[_df['일자'] == pd.to_datetime('2025-07-10')]
predictions = sum(models[k].predict(X_pred) * w for k, w in ensemble_weights.items())
pred_int = int(np.round(predictions[0]))
print(f"2025-07-14 예측 시가 (KRW): {pred_int}원")

# 2025-07-14 예측 시가 (KRW): 844564원