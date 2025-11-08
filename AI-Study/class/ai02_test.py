import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

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

# Select and rename
series_h = df_h[['일자','시가','종가']].rename(columns={'시가':'open_h','종가':'close_h'})
series_s = df_s[['일자','종가']].rename(columns={'종가':'close_s'})

# Align dates
df = pd.merge(series_h, series_s, on='일자', how='inner')

# Create lag features
for lag in range(1, N_LAGS+1):
    df[f'lag_h_{lag}'] = df['close_h'].apply(lambda x: str(x)).replace({',': ''}, regex=True).astype(float).shift(lag)
    df[f'lag_s_{lag}'] = df['close_s'].apply(lambda x: str(x)).replace({',': ''}, regex=True).astype(float).shift(lag)

df['target'] = df['open_h'].apply(lambda x: str(x)).replace({',': ''}, regex=True).astype(float).shift(-1)
df.dropna(inplace=True)

# Split features
features_h = [f'lag_h_{i}' for i in range(1, N_LAGS+1)]
features_s = [f'lag_s_{i}' for i in range(1, N_LAGS+1)]
X_h, X_s = df[features_h], df[features_s]
y = df['target']

# Train cutoff
date_cutoff = pd.to_datetime('2025-07-10')
mask = df['일자'] <= date_cutoff
Xh_train, Xs_train, y_train = X_h[mask], X_s[mask], y[mask]

# -----------------------------
# 2. TimeSeries CV
# -----------------------------
tscv = TimeSeriesSplit(n_splits=3)

def optimize_xgb(X_train, y_train):
    def objective(trial):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'n_estimators': trial.suggest_int('n_estimators',50,300),
            'max_depth': trial.suggest_int('max_depth',3,12),
            'learning_rate': trial.suggest_float('learning_rate',0.01,0.3,log=True),
            'subsample': trial.suggest_float('subsample',0.6,1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.6,1.0),
            'objective':'reg:squarederror',
            'seed':42
        }
        cv_res = xgb.cv(params, dtrain, num_boost_round=params['n_estimators'],
                        nfold=3, early_stopping_rounds=20,
                        metrics='mae', verbose_eval=False)
        return cv_res['test-mae-mean'].min()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    return study.best_params

# Optimize two models
dims_h = optimize_xgb(Xh_train, y_train)
dims_s = optimize_xgb(Xs_train, y_train)

# -----------------------------
# 3. Train final XGB models
# -----------------------------
model_h = XGBRegressor(**dims_h, objective='reg:squarederror', random_state=42)
model_h.fit(Xh_train, y_train)
joblib.dump(model_h, os.path.join(MODEL_DIR, 'model_h.pkl'))

model_s = XGBRegressor(**dims_s, objective='reg:squarederror', random_state=42)
model_s.fit(Xs_train, y_train)
joblib.dump(model_s, os.path.join(MODEL_DIR, 'model_s.pkl'))

# -----------------------------
# 4. Ensemble prediction (1:0)
# -----------------------------
Xh_pred = X_h[df['일자']==pd.to_datetime('2025-07-10')]
Xs_pred = X_s[df['일자']==pd.to_datetime('2025-07-10')]
pred_h = model_h.predict(Xh_pred)
pred_s = model_s.predict(Xs_pred)
# Voting weight: Hanwha 1, Samsung 0
pred_final = pred_h * 1 + pred_s * 0
pred_int = int(np.round(pred_final[0]))
print(f"2025-07-14 예측 시가 (KRW): {pred_int}원")

# 2025-07-14 예측 시가 (KRW): 848184원

# 2025-07-14 예측 시가 (KRW): 844732원

# 2025-07-14 예측 시가 (KRW): 847507원