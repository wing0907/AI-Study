import os
import json
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load-only Inference Script (aligned with training pipeline)
# -----------------------------
N_LAGS = 5
TEST_DIR = r'C:\Study25\_data\test\\'
MODEL_DIR = 'models'

# 1. Load and align data (same as training)
# Load Hanwha and Samsung data
df_h = pd.read_csv(os.path.join(TEST_DIR, '한화에어로스페이스 250711.csv'), encoding='cp949')
df_s = pd.read_csv(os.path.join(TEST_DIR, '삼성전자 250711.csv'), encoding='cp949')
# Parse dates and clean closing prices
for df, col in [(df_h, '종가'), (df_s, '종가')]:
    df['일자'] = pd.to_datetime(df['일자'], format='%Y/%m/%d')
    df[col] = df[col].str.replace(',', '').astype(float)
    df.sort_values('일자', inplace=True)
# Rename columns for merge
series_h = df_h[['일자', '종가']].rename(columns={'종가':'close_h'})
series_s = df_s[['일자', '종가']].rename(columns={'종가':'close_s'})
# Merge on date
df = pd.merge(series_h, series_s, on='일자', how='inner')

# 2. Create lag features exactly as in training
for lag in range(1, N_LAGS+1):
    df[f'lag_h_{lag}'] = df['close_h'].shift(lag)
# Drop NaNs
df.dropna(inplace=True)

# 3. Select features for target date (2025-07-10)
predict_date = pd.to_datetime('2025-07-10')
Xh_pred = df.loc[df['일자'] == predict_date, [f'lag_h_{i}' for i in range(1, N_LAGS+1)]].values

# 4. Load models and weights
model_h = joblib.load(os.path.join(MODEL_DIR, 'model_h_20250711_130235.pkl'))
# Samsung model exists but weight=0, so optional
weights_path = os.path.join(MODEL_DIR, 'ensemble_weights_20250711_130235.json')
if os.path.exists(weights_path):
    with open(weights_path, 'r') as f:
        w = json.load(f)
    weight_h = w.get('hanwha', 1.0)
else:
    weight_h = 1.0

# 5. Predict and output
pred_h = model_h.predict(Xh_pred)
final_pred = pred_h * weight_h
final_int = int(np.round(final_pred[0]))
print(f"2025-07-14 예측 한화시가 (KRW): {final_int}원")

# 2025-07-14 예측 한화시가 (KRW): 847507원