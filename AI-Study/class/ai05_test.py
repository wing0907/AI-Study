import os
import json
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Configuration
N_LAGS = 60
TEST_DIR = r'C:\Study25\_data\test'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
date_suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
cutoff = pd.to_datetime('2025-07-10')

# 1. Load & preprocess data for Hanwha and Samsung
df_h = pd.read_csv(os.path.join(TEST_DIR, '한화에어로스페이스 250711.csv'), encoding='cp949')
df_h['일자'] = pd.to_datetime(df_h['일자'], format='%Y/%m/%d')
df_h['open_h'] = df_h['시가'].str.replace(',', '').astype(float)
df_h['close_h'] = df_h['종가'].str.replace(',', '').astype(float)

df_s = pd.read_csv(os.path.join(TEST_DIR, '삼성전자 250711.csv'), encoding='cp949')
df_s['일자'] = pd.to_datetime(df_s['일자'], format='%Y/%m/%d')
df_s['close_s'] = df_s['종가'].str.replace(',', '').astype(float)

# Merge on date
df = pd.merge(df_h[['일자','open_h','close_h']], df_s[['일자','close_s']], on='일자', how='inner')

# 2. Create rate-of-change (ROC) features for both symbols
roc_h = [f'roc_h_{i}' for i in range(1, N_LAGS+1)]
roc_s = [f'roc_s_{i}' for i in range(1, N_LAGS+1)]
for i in range(1, N_LAGS+1):
    df[f'roc_h_{i}'] = df['close_h'].pct_change(periods=i)
    df[f'roc_s_{i}'] = df['close_s'].pct_change(periods=i)
# Drop rows without full ROC history
df_lag = df.dropna(subset=roc_h + roc_s).reset_index(drop=True)

# 3. Split training and inference sets
df_train = df_lag[df_lag['일자'] <= cutoff].copy()
df_train['target'] = df_train['open_h'].shift(-1)
df_train.dropna(subset=['target'], inplace=True)
Xh_train = df_train[roc_h]
Xs_train = df_train[roc_s]
y_train = df_train['target']

df_pred = df_lag[df_lag['일자'] == cutoff]
if df_pred.empty:
    df_pred = df_lag[df_lag['일자'] <= cutoff].iloc[[-1]]
Xh_pred = df_pred[roc_h].values
Xs_pred = df_pred[roc_s].values

# 4. Hyperparameter tuning for both models using MAE
def optimize_xgb(X, y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'objective': 'reg:squarederror', 'seed': 42
        }
        dtrain = xgb.DMatrix(X, label=y)
        cv_res = xgb.cv(
            params, dtrain,
            num_boost_round=params['n_estimators'], nfold=5,
            early_stopping_rounds=20, metrics='mae',
            seed=42, verbose_eval=False
        )
        return cv_res['test-mae-mean'].min()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    return study.best_params
best_params_h = optimize_xgb(Xh_train, y_train)
best_params_s = optimize_xgb(Xs_train, y_train)

# 5. Train final XGB models & evaluate with RMSE and R²
dm_h = XGBRegressor(**best_params_h, random_state=42)
dm_h.fit(Xh_train, y_train)
preds_h = dm_h.predict(Xh_train)
rmse_h = np.sqrt(mean_squared_error(y_train, preds_h))
r2_h = r2_score(y_train, preds_h)
print(f"Train RMSE (Hanwha): {rmse_h:.2f}원, R²: {r2_h:.3f}")

dm_s = XGBRegressor(**best_params_s, random_state=42)
dm_s.fit(Xs_train, y_train)
preds_s = dm_s.predict(Xs_train)
rmse_s = np.sqrt(mean_squared_error(y_train, preds_s))
r2_s = r2_score(y_train, preds_s)
print(f"Train RMSE (Samsung): {rmse_s:.2f}원, R²: {r2_s:.3f}")

# 6. Save models & ensemble weights
h_path = os.path.join(MODEL_DIR, f'model_h_{date_suffix}.pkl')
joblib.dump(dm_h, h_path)
s_path = os.path.join(MODEL_DIR, f'model_s_{date_suffix}.pkl')
joblib.dump(dm_s, s_path)
weights = {'hanwha': 1.0, 'samsung': 0.0}
w_path = os.path.join(MODEL_DIR, f'ensemble_weights_{date_suffix}.json')
with open(w_path, 'w') as f:
    json.dump(weights, f)
print('Saved:', h_path, s_path, w_path)

# 7. Predict next open using ensemble
pred_h = dm_h.predict(Xh_pred)
pred_s = dm_s.predict(Xs_pred)
pred_final = pred_h * weights['hanwha'] + pred_s * weights['samsung']
pred_int = int(np.round(pred_final[0]))
print(f"2025-07-14 예측 시가: {pred_int}원")
