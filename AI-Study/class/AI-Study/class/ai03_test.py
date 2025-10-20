import os
import json
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import random

# -----------------------------
# Configuration
# -----------------------------
N_LAGS = 60
TEST_DIR = r'C:\Study25\_data\test'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

date_suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# -----------------------------
# 1. Load and preprocess data
# -----------------------------
# Load CSVs
df_h = pd.read_csv(os.path.join(TEST_DIR, '한화에어로스페이스 250711.csv'), encoding='cp949')
df_s = pd.read_csv(os.path.join(TEST_DIR, '삼성전자 250711.csv'), encoding='cp949')

# Clean and convert types for df_h (시가, 종가)
df_h['일자'] = pd.to_datetime(df_h['일자'], format='%Y/%m/%d')
df_h['시가'] = df_h['시가'].str.replace(',', '').astype(float)
df_h['종가'] = df_h['종가'].str.replace(',', '').astype(float)
df_h.sort_values('일자', inplace=True)

# Clean and convert for df_s (종가)
df_s['일자'] = pd.to_datetime(df_s['일자'], format='%Y/%m/%d')
df_s['종가'] = df_s['종가'].str.replace(',', '').astype(float)
df_s.sort_values('일자', inplace=True)

# Rename and merge
df_h = df_h[['일자', '시가', '종가']].rename(columns={'시가':'open_h','종가':'close_h'})
df_s = df_s[['일자', '종가']].rename(columns={'종가':'close_s'})
df = pd.merge(df_h, df_s, on='일자', how='inner')

# Create lag features
for lag in range(1, N_LAGS+1):
    df[f'lag_h_{lag}'] = df['close_h'].shift(lag)
    df[f'lag_s_{lag}'] = df['close_s'].shift(lag)

# Target: next day open_h
df['target'] = df['open_h'].shift(-1)
df.dropna(inplace=True)

# Feature matrices and target vector
features_h = [f'lag_h_{i}' for i in range(1, N_LAGS+1)]
features_s = [f'lag_s_{i}' for i in range(1, N_LAGS+1)]
X_h, X_s = df[features_h], df[features_s]
y = df['target']

# Train cutoff mask
cutoff = pd.to_datetime('2025-07-10')
mask = df['일자'] <= cutoff
Xh_train, Xs_train, y_train = X_h[mask], X_s[mask], y[mask]

# -----------------------------
# 2. TimeSeries CV
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)


seed = 333
random.seed(seed)
np.random.seed(seed)
# -----------------------------
# 3. HYPERPARAMETER TUNING (Optuna + xgb.cv) ### HIGHLIGHT ###
# Search for best hyperparams minimizing MAE via CV
# -----------------------------
def optimize_xgb(X_train, y_train):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'objective': 'reg:squarederror',
            'seed': seed
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        cv_res = xgb.cv(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            nfold=5,
            early_stopping_rounds=20,
            metrics='mae',
            seed=seed,
            verbose_eval=False
        )
        return cv_res['test-mae-mean'].min()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    return study.best_params

best_params_h = optimize_xgb(Xh_train, y_train)
best_params_s = optimize_xgb(Xs_train, y_train)

# -----------------------------
# 4. Train final XGB models & save with timestamp
# -----------------------------
model_h = XGBRegressor(**best_params_h, objective='reg:squarederror', random_state=42)
model_h.fit(Xh_train, y_train)
h_path = os.path.join(MODEL_DIR, f'model_h_{date_suffix}.pkl')
joblib.dump(model_h, h_path)

model_s = XGBRegressor(**best_params_s, objective='reg:squarederror', random_state=42)
model_s.fit(Xs_train, y_train)
s_path = os.path.join(MODEL_DIR, f'model_s_{date_suffix}.pkl')
joblib.dump(model_s, s_path)

# Save ensemble weights
ensemble_weights = {'hanwha': 1.0, 'samsung': 0.0}
w_path = os.path.join(MODEL_DIR, f'ensemble_weights_{date_suffix}.json')
with open(w_path, 'w') as f:
    json.dump(ensemble_weights, f)

print('Models saved:', h_path, s_path)
print('Weights saved:', w_path)

# -----------------------------
# 5. Ensemble prediction (1:0)
# -----------------------------
Xh_pred = X_h[df['일자'] == cutoff]
pred_h = model_h.predict(Xh_pred)
pred_s = model_s.predict(X_s[df['일자'] == cutoff])
pred_final = pred_h * ensemble_weights['hanwha'] + pred_s * ensemble_weights['samsung']
pred_int = int(np.round(pred_final[0]))
print(f"2025-07-14 예측 시가 (KRW): {pred_int}원")
print('SEED : ', seed)

# 2025-07-14 예측 시가 (KRW): 840029원


# Models saved: models\model_h_20250711_131212.pkl models\model_s_20250711_131212.pkl
# Weights saved: models\ensemble_weights_20250711_131212.json
# 2025-07-14 예측 시가 (KRW): 843131원


# Models saved: models\model_h_20250711_131654.pkl models\model_s_20250711_131654.pkl
# Weights saved: models\ensemble_weights_20250711_131654.json
# 2025-07-14 예측 시가 (KRW): 837424원


# Models saved: models\model_h_20250711_131757.pkl models\model_s_20250711_131757.pkl
# Weights saved: models\ensemble_weights_20250711_131757.json
# 2025-07-14 예측 시가 (KRW): 848525원


# Models saved: models\model_h_20250711_150925.pkl models\model_s_20250711_150925.pkl
# Weights saved: models\ensemble_weights_20250711_150925.json
# 2025-07-14 예측 시가 (KRW): 847342원
# SEED :  222

# Models saved: models\model_h_20250711_171628.pkl models\model_s_20250711_171628.pkl
# Weights saved: models\ensemble_weights_20250711_171628.json
# 2025-07-14 예측 시가 (KRW): 847167원
# SEED :  222

# Models saved: models\model_h_20250711_182834.pkl models\model_s_20250711_182834.pkl
# Weights saved: models\ensemble_weights_20250711_182834.json
# 2025-07-14 예측 시가 (KRW): 847268원
# SEED :  519


# Models saved: models\model_h_20250711_183706.pkl models\model_s_20250711_183706.pkl
# Weights saved: models\ensemble_weights_20250711_183706.json
# 2025-07-14 예측 시가 (KRW): 847292원
# SEED :  333