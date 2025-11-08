import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path = './_data/kaggle/jena/'
save_path = './_save/jena/'
os.makedirs(save_path, exist_ok=True)

# ----------------------
# 데이터 로딩 및 전처리
# ----------------------
df = pd.read_csv(path + 'jena_climate_2009_2016.csv')
df['datetime'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
df = df.drop(columns=['Date Time'])

# ✅ 이상치 조정
clip_settings = {
    'wv (m/s)': (0.1, 50),
    'T (degC)': (-40, 50),
    'p (mbar)': (850, 1100),
    'wd (deg)': (0, 360)
}
for col, (low, high) in clip_settings.items():
    df[col] = df[col].clip(lower=low, upper=high)

# 시간 관련 파생 변수 추가
df['hour'] = df['datetime'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# ✅ 풍향을 각도로 변환
wd_rad = np.deg2rad(df['wd (deg)'])
df['wd_sin'] = np.sin(wd_rad)
df['wd_cos'] = np.cos(wd_rad)

# ✅ 사용 컬럼 축소 및 결측치 처리
drop_columns = ['wd (deg)', 'datetime', 'Tpot (K)', 'Tdew (degC)', 'p (mbar)', 'rh (%)', 'hour']
x_columns = df.columns.difference(drop_columns + ['wd_sin', 'wd_cos'])
y_columns = ['wd_sin', 'wd_cos']

x = df[x_columns].copy()
x['missing_flag'] = x.isnull().any(axis=1).astype(int)
x = x.interpolate(method='linear', limit_direction='both')
x = x.reindex(columns=list(x_columns) + ['missing_flag'])

y = df[y_columns].copy()
y = y.interpolate(method='linear', limit_direction='both')

# ----------------------
# 예측 구간 정의 및 학습 데이터 절단
# ----------------------
y_start_time = pd.Timestamp('2016-12-31 00:10:00')
y_end_time = pd.Timestamp('2017-01-01 00:00:00')
x_input_start_time = y_start_time - pd.Timedelta(minutes=10 * 144)
x_input_end_time = y_start_time - pd.Timedelta(minutes=10)

# x와 y에서 예측 타깃 구간 제거
y_range = (df['datetime'] >= y_start_time) & (df['datetime'] <= y_end_time)
x_range = df['datetime'] < y_start_time

x_train = x.loc[x_range]
y_train = y.loc[x_range]

# ----------------------
# 시계열 데이터 슬라이싱 + stride
# ----------------------
def split_xy_sequences(x_data, y_data, input_window=144, output_window=144, stride=6):
    x_seq, y_seq = [], []
    for i in range(0, len(x_data) - input_window - output_window + 1, stride):
        x_subset = x_data.iloc[i : i + input_window].values
        y_subset = y_data.iloc[i + input_window : i + input_window + output_window].values
        x_seq.append(x_subset)
        y_seq.append(y_subset)
    return np.array(x_seq), np.array(y_seq)

x_sequences, y_sequences = split_xy_sequences(x_train, y_train, input_window=144, output_window=144, stride=6)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler(feature_range=(-1, 1))

num_features = x_sequences.shape[2]
x_sequences_2d = x_sequences.reshape(-1, num_features)
y_sequences_2d = y_sequences.reshape(-1, 2)

x_scaled = x_scaler.fit_transform(x_sequences_2d).reshape(x_sequences.shape)
y_scaled = y_scaler.fit_transform(y_sequences_2d).reshape(y_sequences.shape)

# ----------------------
# 예측용 입력 데이터 준비
# ----------------------
x_input_raw = df.loc[(df['datetime'] >= x_input_start_time) & (df['datetime'] <= x_input_end_time), x_columns].copy()
x_input_raw['missing_flag'] = x_input_raw.isnull().any(axis=1).astype(int)
x_input_raw = x_input_raw.interpolate(method='linear', limit_direction='both')
x_input_raw = x_input_raw.reindex(columns=list(x_columns) + ['missing_flag'])

if len(x_input_raw.columns) != num_features:
    raise ValueError(f"입력 피처 수 불일치: {len(x_input_raw.columns)}개, 예상: {num_features}개")

if len(x_input_raw) < 144:
    raise ValueError(f"슬라이싱된 데이터가 144개보다 작습니다. 현재: {len(x_input_raw)}개")

x_input_scaled = x_scaler.transform(x_input_raw.values).reshape(1, 144, num_features)

# ----------------------
# 모델 구성 및 가중치 불러오기
# ----------------------
model = Sequential([
    Bidirectional(GRU(128, return_sequences=True), input_shape=(144, num_features)),
    Dropout(0.2),
    GRU(64),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(288, activation=None)
])

model.compile(optimizer='adam', loss='mse')

# ✅ 저장된 가중치 불러오기
weights_path = os.path.join(save_path, 'jena_weights.hdf5')
model.load_weights(weights_path)

# ----------------------
# 예측 및 평가
# ----------------------
y_true_df = df.loc[y_range, ['datetime', 'wd (deg)']]
y_pred_scaled = model.predict(x_input_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 2))

wd_pred_rad = np.arctan2(y_pred[:, 1], y_pred[:, 0])
wd_pred_deg = (np.rad2deg(wd_pred_rad) + 360) % 360

y_true = y_true_df['wd (deg)'].values
rmse = np.sqrt(mean_squared_error(y_true, wd_pred_deg))
print(f'✅ 저장된 가중치로 예측한 RMSE: {rmse}')

# 제출 저장
submit = pd.DataFrame({
    'datetime': y_true_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').values,
    'wd(deg)_true': y_true,
    'wd(deg)_prediction': wd_pred_deg
})
submit.to_csv(save_path + 'jena_장우진_submit_loaded.csv', index=False)
print(f'📁 예측 결과 저장 완료: {save_path + "jena_장우진_submit_loaded.csv"}')
