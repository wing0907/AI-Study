# Conv1D로 RNN을 이겨라!!!

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

path = './_data/kaggle/jena/'

# ----------------------
# 데이터 로딩 및 전처리
# ----------------------
df = pd.read_csv(path + 'jena_cleaned_final.csv')

# ✅ datetime 컬럼 제거 버전 (이 부분 삭제)
# df['datetime'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
# df = df.drop(columns=['Date Time'])

# ✅ 이상치 조정 (삭제하지 않고 클리핑)
df['wv (m/s)'] = df['wv (m/s)'].clip(lower=0.1, upper=50)
df['T (degC)'] = df['T (degC)'].clip(lower=-40, upper=50)
df['p (mbar)'] = df['p (mbar)'].clip(lower=850, upper=1100)
df['wd (deg)'] = df['wd (deg)'].clip(lower=0, upper=360)

# ✅ 시간 파생 변수 제거
# df['hour'] = df['datetime'].dt.hour
# df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
# df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

x_columns = df.columns.difference(['wd (deg)'])  # datetime 제거됨
y_column = 'wd (deg)'

# ✅ 예측 대상 구간 수동 설정 (없으면 슬라이싱 기반 처리만 가능)
forecast_start_index = -144  # 마지막 1일치
forecast_df = df.iloc[forecast_start_index:]

# ----------------------
# 데이터 준비
# ----------------------
timesteps = 144
train_df = df.iloc[:forecast_start_index]
train_df_recent = train_df

print(train_df_recent[x_columns].dtypes)
x_columns = df.select_dtypes(include=[np.number]).columns.difference(['wd (deg)'])

x_scaler = MinMaxScaler()
x_scaled = x_scaler.fit_transform(train_df_recent[x_columns])


y_deg = train_df_recent[y_column].values
y_rad = np.deg2rad(y_deg)
y_wv = train_df_recent['wv (m/s)'].values
y_u = y_wv * np.cos(y_rad)
y_v = y_wv * np.sin(y_rad)

scaled_df = pd.DataFrame(x_scaled, columns=x_columns)
scaled_df['u'] = y_u
scaled_df['v'] = y_v

# ----------------------
# 시계열 데이터 분할
# ----------------------
def split_xy(dataset, timesteps):
    x, y = [], []
    for i in range(len(dataset) - timesteps):
        subset = dataset[i: i + timesteps]
        x.append(subset[:, :-2])
        y.append(subset[:, -2:])
    return np.array(x), np.array(y)

data = scaled_df.to_numpy()
x, y = split_xy(data, timesteps)

# ----------------------
# 예측 루프용 데이터
# ----------------------
x_input_raw = df.iloc[forecast_start_index - timesteps:forecast_start_index][x_columns]
x_input_scaled = x_scaler.transform(x_input_raw)
x_input = x_input_scaled.reshape(1, timesteps, len(x_columns))

# ----------------------
# 모델 구성
# ----------------------
model = Sequential([
    Conv1D(128, 3, input_shape=(timesteps, len(x_columns))),
    Dropout(0.2),
    Conv1D(128, 2),
    Dropout(0.2),
    Conv1D(128, 2),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(288, activation=None),
])

model.compile(optimizer='adam', loss='mse')

# ----------------------
# 학습
# ----------------------
save_dir = r'C:/Study25/_save/jena'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'jena_best_model_nodate.hdf5')
weights_path = os.path.join(save_dir, 'jena_weights_nodate.hdf5')

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mc = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)

model.fit(x, y.reshape(x.shape[0], -1), epochs=100, batch_size=128, validation_split=0.2, callbacks=[es, mc])

# ----------------------
# 가중치 저장
# ----------------------
model.save_weights(weights_path)

# ----------------------
# 예측
# ----------------------
pred = model.predict(x_input, verbose=0)
y_uv_pred = pred.reshape(144, 2)
y_pred_rad = np.arctan2(y_uv_pred[:, 1], y_uv_pred[:, 0])
y_pred_deg = np.rad2deg(y_pred_rad)
y_pred_deg = (y_pred_deg + 360) % 360

# ----------------------
# 평가
# ----------------------
y_true_deg = forecast_df[y_column].values
y_true_rad = np.deg2rad(y_true_deg)
y_true_u = forecast_df['wv (m/s)'].values * np.cos(y_true_rad)
y_true_v = forecast_df['wv (m/s)'].values * np.sin(y_true_rad)

u_rmse = np.sqrt(mean_squared_error(y_true_u, y_uv_pred[:, 0]))
v_rmse = np.sqrt(mean_squared_error(y_true_v, y_uv_pred[:, 1]))
total_rmse = np.sqrt(u_rmse**2 + v_rmse**2)
print("✅ u-v 벡터 기반 RMSE:", total_rmse)

# ----------------------
# 제출 저장
# ----------------------
submit_df = pd.DataFrame({
    'index': np.arange(len(y_pred_deg)),
    'wd (deg)': y_pred_deg
})
submit_path = os.path.join(save_dir, 'jena_nodate_submit.csv')
submit_df.to_csv(submit_path, index=False)
print("📁제출 파일 저장 완료:", submit_path)

# ✅ u-v 벡터 기반 RMSE: 0.7154899793545809
# 📁제출 파일 저장 완료: C:/Study25/_save/jena\jena_nodate_submit.csv