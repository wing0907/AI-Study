"""
기존 데이터에서 y를 144개(1일치)를 맞춰야하므로,
y데이터를 위로 144개 shift해서
새로운 행렬 데이터를 만들어서
144개를 예측한다.

"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

path = './_data/kaggle/jena/'

# ----------------------
# 데이터 로딩 및 전처리
# ----------------------
df = pd.read_csv(path + 'jena_climate_2009_2016.csv')
df['datetime'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
df = df.drop(columns=['Date Time'])

# ✅ 이상치 조정 (삭제하지 않고 클리핑)
df['wv (m/s)'] = df['wv (m/s)'].clip(lower=0.1, upper=50)
df['T (degC)'] = df['T (degC)'].clip(lower=-40, upper=50)
df['p (mbar)'] = df['p (mbar)'].clip(lower=850, upper=1100)
df['wd (deg)'] = df['wd (deg)'].clip(lower=0, upper=360)

# 시간 관련 파생 변수 추가
df['hour'] = df['datetime'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# ✅ 사용 컬럼 축소: 중요 변수만 선택
drop_columns = ['Tpot (K)', 'Tdew (degC)', 'p (mbar)', 'rh (%)', 'hour']
x_columns = df.columns.difference(['wd (deg)', 'datetime'] + drop_columns)
y_column = 'wd (deg)'

forecast_start = pd.Timestamp('2016-12-31 00:10:00')
forecast_end = pd.Timestamp('2017-01-01 00:00:00')

matching = df[df['datetime'] >= forecast_start]
if matching.empty:
    raise ValueError("forecast_start 이후 데이터가 없습니다.")
forecast_start = matching['datetime'].iloc[0]
forecast_start_index = matching.index[0]
forecast_df = df[(df['datetime'] >= forecast_start) & (df['datetime'] <= forecast_end)]

# ----------------------
# 데이터 준비 및 시계열 분할
# ----------------------
def split_xy_multistep(x_data, y_series, timesteps=144, out_steps=144):
    x, y = [], []
    for i in range(len(x_data) - timesteps - out_steps + 1):
        x_end = i + timesteps
        y_end = x_end + out_steps
        x_seq = x_data[i:x_end]
        y_seq = y_series[x_end:y_end]
        x.append(x_seq)
        y.append(y_seq)
    return np.array(x), np.array(y)

x_scaled = MinMaxScaler().fit_transform(df[x_columns])
y_deg = df[y_column].values

x, y = split_xy_multistep(x_scaled, y_deg, timesteps=144, out_steps=144)

x = x.reshape(x.shape[0], -1)  # (n, 144 * features)
y = y.reshape(y.shape[0], -1)  # (n, 144)

x_input_raw = df.iloc[forecast_start_index - 144:forecast_start_index][x_columns]
x_input_scaled = MinMaxScaler().fit(x_scaled).transform(x_input_raw)
x_input = x_input_scaled.reshape(1, -1)

# ----------------------
# 모델 구성
# ----------------------
model = Sequential([
    Dense(128, activation='relu', input_shape=(x.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation=None),
])

model.compile(optimizer='adam', loss='mse')

# ----------------------
# 학습
# ----------------------
save_dir = r'C:/Study25/_save/jena'
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'jena_best_model_multistep.hdf5')
weights_path = os.path.join(save_dir, 'jena_weights_multistep.hdf5')

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mc = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)

model.fit(x, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es, mc])

# ----------------------
# 가중치 저장
# ----------------------
model.save_weights(weights_path)

# ----------------------
# 예측 및 제출
# ----------------------
pred = model.predict(x_input, verbose=0)
y_pred_deg = pred.flatten()  # shape: (144,)

submit_df = pd.DataFrame({
    'datetime': forecast_df['datetime'].reset_index(drop=True),
    'wd (deg)': y_pred_deg
})
submit_path = os.path.join(save_dir, 'jena_multistep_submit.csv')
submit_df.to_csv(submit_path, index=False)
print("📁제출 파일 저장 완료:", submit_path)
