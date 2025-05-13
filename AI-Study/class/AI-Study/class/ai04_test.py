import os
import json
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Configuration
N_LAGS = 30
test_dir = r'C:\Study25\_data\test'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
date_suffix = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
cutoff = pd.to_datetime('2025-07-10')

# 1. Load & preprocess data
df_h = pd.read_csv(os.path.join(test_dir, '한화에어로스페이스 250711.csv'), encoding='cp949')
df_s = pd.read_csv(os.path.join(test_dir, '삼성전자 250711.csv'), encoding='cp949')
# Clean Hanwha
df_h['일자'] = pd.to_datetime(df_h['일자'], format='%Y/%m/%d')
df_h['open_h'] = df_h['시가'].str.replace(',', '').astype(float)
df_h['close_h'] = df_h['종가'].str.replace(',', '').astype(float)
df_h = df_h[['일자','open_h','close_h']]
# Clean Samsung
df_s['일자'] = pd.to_datetime(df_s['일자'], format='%Y/%m/%d')
df_s['close_s'] = df_s['종가'].str.replace(',', '').astype(float)
df_s = df_s[['일자','close_s']]
# Merge datasets
df = pd.merge(df_h, df_s, on='일자', how='inner')

# 2. Create lag features
features_h = [f'lag_h_{i}' for i in range(1, N_LAGS+1)]
features_s = [f'lag_s_{i}' for i in range(1, N_LAGS+1)]
for i in range(1, N_LAGS+1):
    df[f'lag_h_{i}'] = df['close_h'].shift(i)
    df[f'lag_s_{i}'] = df['close_s'].shift(i)
# Drop rows with missing lag features
df_lag = df.dropna(subset=features_h + features_s).reset_index(drop=True)

# 3. Split training and inference sets
# Training: rows with date <= cutoff and next-day target available
df_train = df_lag[df_lag['일자'] <= cutoff].copy()
# Create target for training
df_train['target'] = df_train['open_h'].shift(-1)
df_train = df_train.dropna(subset=['target'])
Xh_train = df_train[features_h]; Xs_train = df_train[features_s]; y_train = df_train['target']

df_inf = df_lag[df_lag['일자'] == cutoff]
if df_inf.empty:
    # 휴장일 등으로 데이터가 없으면 마지막 가용 행 사용
    df_inf = df_lag.iloc[[-1]]
Xh_pred = df_inf[features_h].values
Xs_pred = df_inf[features_s].values

# 4. TimeSeries CV setup
tscv = TimeSeriesSplit(n_splits=5)

# 5. Hyperparameter tuning using RMSE
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
        cv = xgb.cv(params, dtrain,
                    num_boost_round=params['n_estimators'], nfold=5,
                    early_stopping_rounds=20, metrics='rmse',
                    seed=42, verbose_eval=False)
        return cv['test-rmse-mean'].min()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    return study.best_params
best_params_h = optimize_xgb(Xh_train, y_train)
best_params_s = optimize_xgb(Xs_train, y_train)

# 6. Train final models & evaluate training performance
model_h = XGBRegressor(**best_params_h, objective='reg:squarederror', random_state=42)
model_h.fit(Xh_train, y_train)
preds_h_train = model_h.predict(Xh_train)
rmse_h = np.sqrt(mean_squared_error(y_train, preds_h_train))
r2_h = r2_score(y_train, preds_h_train)
print(f"Train RMSE (Hanwha): {rmse_h:.2f}원, R²: {r2_h:.3f}")

model_s = XGBRegressor(**best_params_s, objective='reg:squarederror', random_state=42)
model_s.fit(Xs_train, y_train)
preds_s_train = model_s.predict(Xs_train)
rmse_s = np.sqrt(mean_squared_error(y_train, preds_s_train))
r2_s = r2_score(y_train, preds_s_train)
print(f"Train RMSE (Samsung): {rmse_s:.2f}원, R²: {r2_s:.3f}")

# 7. Save models & weights
h_path = os.path.join(model_dir, f'model_h_{date_suffix}.pkl')
joblib.dump(model_h, h_path)
s_path = os.path.join(model_dir, f'model_s_{date_suffix}.pkl')
joblib.dump(model_s, s_path)
weights = {'hanwha':1.0, 'samsung':0.0}
w_path = os.path.join(model_dir, f'ensemble_weights_{date_suffix}.json')
with open(w_path, 'w') as f: json.dump(weights, f)
print('Saved:', h_path, s_path, w_path)

# 8. Predict next open
pred_h = model_h.predict(Xh_pred)
pred_s = model_s.predict(Xs_pred)
pred_final = pred_h * weights['hanwha'] + pred_s * weights['samsung']
pred_int = int(np.round(pred_final[0]))
print(f"2025-07-14 예측 시가 (KRW): {pred_int}원")


# Train RMSE (Hanwha): 1863.51원, R²: 1.000
# Train RMSE (Samsung): 3891.48원, R²: 0.999
# Saved: models\model_h_20250711_175607.pkl models\model_s_20250711_175607.pkl models\ensemble_weights_20250711_175607.json
# 2025-07-14 예측 시가 (KRW): 52054원
