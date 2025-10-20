import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
path = 'C:\study25jun\_data\kaggle\jena_clime\\'
train_csv = pd.read_csv(path+'jena_climate_2009_2016.csv') #"T (degC)"
#2016.12.31 'O' 144개 찾기 

# -----------------------
# 1. 데이터 전처리
# -----------------------

# 기본 세팅
timesteps = 24
target_horizon = 144
stride = 1

y_test_1 = train_csv[ (train_csv['Date Time'].str.contains("31.12.2016", regex=False)) &
    (train_csv['Date Time'] != "31.12.2016 00:00:00")]['wd (deg)'].copy()
y_test_2 = train_csv[train_csv['Date Time'].str.contains("01.01.2017")]['wd (deg)'].copy()
y_test = pd.concat([y_test_1, y_test_2], ignore_index=True) #ignore_index 인덱스를 새로 매겨줘

# wind direction → sin/cos 변환
train_csv['wd_sin'] = np.sin(np.radians(train_csv['wd (deg)']))
train_csv['wd_cos'] = np.cos(np.radians(train_csv['wd (deg)']))

# 날짜 처리
train_csv['Date Time'] = pd.to_datetime(train_csv['Date Time'])
train_csv['hour'] = train_csv['Date Time'].dt.hour
train_csv['month'] = train_csv['Date Time'].dt.month
train_csv['day'] = train_csv['Date Time'].dt.day
train_csv['minute'] = train_csv['Date Time'].dt.minute
train_csv['weekday'] = train_csv['Date Time'].dt.weekday

# 정규화
cols = ['VPmax (mbar)',
        'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)',
        'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)'
        ]
# 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)',
#         'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)',
#         'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)'
# scalers = {}
# for col in cols:
#     scaler = StandardScaler()
#     train_csv[col] = scaler.fit_transform(train_csv[[col]])
#     scalers[col] = scaler
scaler = StandardScaler()
train_csv[cols] = scaler.fit_transform(train_csv[cols])
# x/y raw 배열 생성
# x_raw = train_csv.drop(['Date Time', 'wd (deg)', 'wd_sin', 'wd_cos','Tpot (K)','Tdew (degC)','rh (%)','VPmax (mbar)','VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)',], axis=1).values

x_raw = train_csv[['VPmax (mbar)',
        'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)',
        'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)']].values
y_raw = train_csv[['wd_sin', 'wd_cos']].values

# 미래 예측 제거
x_raw = x_raw[: - (timesteps + 2*target_horizon - 1)]
y_raw = y_raw[: - (timesteps + target_horizon - 1)]


# 슬라이딩 윈도우 함수
def split_xy_stride(x, y, window_size, stride):
    x_seqs, y_seqs = [], []
    for i in range(0, len(x) - window_size + 1, stride):
        x_seq = x[i:i + window_size]
        y_seq = y[i + window_size + target_horizon - 1]
        x_seqs.append(x_seq)
        y_seqs.append(y_seq)
    return np.array(x_seqs), np.array(y_seqs)

x, y = split_xy_stride(x_raw, y_raw, timesteps, stride)

# print(x.shape)
# exit()
# train/val split
# split = int(len(x) * 0.4)
# x_train, x_val = x[split:], x[:split]
# y_train, y_val = y[split:], y[:split]

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.1, random_state= 190)


# -----------------------
# 2. 모델 정의
# -----------------------

# cosine loss
def cosine_loss(y_true, y_pred):
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    return 1 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=-1))

# 모델 구조
model = Sequential([
    LSTM(100, input_shape=(timesteps, x.shape[2]), return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu' ),
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation='relu' ),
    BatchNormalization(),
    Dropout(0.3),
    Dense(2, activation='tanh')  # 최종 출력 sin/cos 값용
])

model.compile(loss=cosine_loss,  optimizer=Adam(clipvalue=1.0))

# -----------------------
# 3. 학습
# -----------------------
filename = 'Keras56_6.hdf7'
es = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only= True,
    filepath=path+filename
)
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=100,
                    batch_size=64,
                    callbacks=[es,mcp],
                    verbose=1)

# -----------------------
# 4. 예측 및 RMSE 평가
# -----------------------

# 테스트셋 준비
required_rows = timesteps + target_horizon - 1
x_test_source = train_csv[-(target_horizon + required_rows):-target_horizon]
# x_test_values = x_test_source.drop(['Date Time', 'wd (deg)', 'wd_sin', 'wd_cos','Tpot (K)','Tdew (degC)','rh (%)','VPmax (mbar)','VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)'], axis=1).values
x_test_values = x_test_source[['VPmax (mbar)',
        'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)',
        'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)']].values
x_test = np.array([x_test_values[i:i+timesteps] for i in range(len(x_test_values) - timesteps + 1)])
# 
# 예측
y_pred = model.predict(x_test)

print(y_pred)

# 벡터 → 각도 변환
pred_angle = np.degrees(np.arctan2(y_pred[:, 0], y_pred[:, 1])) % 360
true_angle = y_test.values  # 이미 저장된 y_test가 각도라면 그대로 사용

# RMSE 계산
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print('✅ RMSE:', rmse(true_angle, pred_angle))
date_times = train_csv[
    (
    (train_csv['Date Time'].dt.date == pd.to_datetime("2016-12-31").date()) |
    (train_csv['Date Time'].dt.date == pd.to_datetime("2017-01-01").date())
    ) & (train_csv['Date Time'] != "2016-12-31 00:00:00")
]['Date Time'].copy()
date_times = date_times.reset_index(drop=True)  # 인덱스 초기화

# y_pred가 numpy array라면 Series로 변환
y_pred_series = pd.Series(pred_angle.flatten(), name='wd (deg)')

# 3. 최종 결과 합치기
submission_df = pd.DataFrame({
    'Date Time': date_times,
    'wd (deg)': y_pred_series
})

# 4. CSV로 저장
submission_df.to_csv(path+"jena_홍석준_submit7.csv", index=False)

# x_test = np.array([
#     x_test_values[i:i+timesteps]
#     for i in range(0, len(x_test_values) - timesteps + 1, stride)
# ])

# ✅ RMSE: 55.12944209732996 3번의 가중치랑 결과