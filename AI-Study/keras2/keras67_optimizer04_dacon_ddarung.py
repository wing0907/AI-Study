# Keras31_gpu_test02_california 복붙
import sklearn as sk
#print(sk.__version__) #1.1.3
import tensorflow as tf
#print(tf.__version__) #2.9.3

from keras.models import Sequential
import numpy as np
import pandas as pd # 전처리 
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.metrics import r2_score
import time
import tensorflow as tf

#1. 데이터
path = 'C:\Study25\_data\dacon\따릉이\\'

                # [=] = b 를 a에 넣어줘 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # . = 현재위치, / = 하위폴더
print(train_csv)                  # [1459 rows x 11 columns] -> [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)                   # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission_csv)             # [715 rows x 1 columns]

print(train_csv.shape)            #(1459, 10)
print(test_csv.shape)             #(715, 9)
print(submission_csv.shape)       #(715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')           # Nan = 결측치              이상치 ex. 서울의 온도 41도

print(train_csv.info())           # 결측치 확인

print(train_csv.describe())

########################   결측치 처리 1. 삭제   ######################
# print(train_csv.isnull().sum()) # 결측치의 개수 출력
print(train_csv.isna().sum())     # 위 함수와 똑같음

train_csv = train_csv.dropna()  #결측치 처리를 삭제하고 남은 값을 반환해 줌
print(train_csv.isna().sum()) 
print(train_csv.info())         # 결측치 확인
print(train_csv)                # [1328 rows x 10 columns]

########################   결측치 처리 2. 평균값 넣기   ######################
# train_csv = train_csv.fillna(train_csv.mean())
# print(train_csv.isna().sum()) 
# print(train_csv.info()) 

########################   테스트 데이터의 결측치 확인   ######################
print(test_csv.info())            # test 데이터에 결측치가 있으면 절대 삭제하지 말 것!
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())            # 715 non-null

#====================== x 와 y 데이터를 나눠준다 =========================#
x = train_csv.drop(['count'], axis=1)    # pandas data framework 에서 행이나 열을 삭제할 수 있다
                #  count라는 axis=1 열 삭제, 참고로 행 삭제는 axis=0
print(x)                                 # [1459 rows x 9 columns]
y = train_csv['count']                   # count 컬럼만 빼서 y 에 넣겠다
print(y.shape)                           #(1459,)



x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size= 0.2,random_state= 36) #6, 21, 36
#print(dataset.info())

best_optim = ''
best_lr = 0
best_r2 = 0
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop
optim = [Adam, Adagrad,SGD,RMSprop]
lr = [0.1,0.01,0.05,0.001,0.0001]
# === 1. 하이퍼파라미터 탐색 루프 ===
best_opt_cls = None
best_lr      = None
best_r2      = -float("inf")

for op in [Adam, Adagrad, SGD, RMSprop]:
    for lr in [0.1, 0.01, 0.05, 0.001, 0.0001]:
        # (1) 모델 생성
        model = Sequential([
            Dense(400, input_dim=9, activation='relu'),
            Dense(400, activation='relu'),
            Dense(300, activation='relu'),
            Dense(300, activation='relu'),
            Dense(300, activation='relu'),
            Dense(50,  activation='relu'),
            Dense(20,  activation='relu'),
            Dense(1)
        ])
        # (2) 컴파일
        model.compile(optimizer=op(learning_rate=lr),
                      loss='mse',
                      metrics=['mae'])
        # (3) 학습 (예: 1 epoch)
        start = time.time()
        model.fit(x_train, y_train,
                  epochs=1,
                  batch_size=32,
                  verbose=0,
                  validation_split=0.1)
        end = time.time()
        
        # (4) 평가
        y_pred = model.predict(x_test)
        if np.isnan(y_pred).any() or np.isinf(y_pred).any():
            print(f"Skipping {op.__name__}, lr={lr} due to invalid preds")
            continue
        r2 = r2_score(y_test, y_pred) 
        # (5) 최적화
        if r2 > best_r2:
            best_r2      = r2
            best_opt_cls = op    # ← 클래스 자체 저장
            best_lr      = lr

print(f"▶ Best: {best_opt_cls.__name__} with lr={best_lr}, R2={best_r2:.4f}")

    

print('best_r2:',r2,'best_optim:',best_optim,"best_lr:",best_lr)
print("걸린시간 :",end-start)


# --- 4. 최종 평가 및 예측 ---

from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop
from sklearn.metrics import r2_score

# 4-1) 옵티마이저 맵
opt_map = {
    'Adam':    Adam,
    'Adagrad': Adagrad,
    'SGD':     SGD,
    'RMSprop': RMSprop
}
# best_optim, best_lr, best_r2 는 이전 루프에서 결정된 값

# === 4. 최종 모델 재생성·학습·평가·제출 ===

# (1) 모델 구조 정의
model = Sequential([
    Dense(400, input_dim=9, activation='relu'),
    Dense(400, activation='relu'),
    Dense(300, activation='relu'),
    Dense(300, activation='relu'),
    Dense(300, activation='relu'),
    Dense(50,  activation='relu'),
    Dense(20,  activation='relu'),
    Dense(1)
])

# (2) 최적 옵티마이저·학습률로 컴파일
model.compile(
    optimizer=best_opt_cls(learning_rate=best_lr),
    loss='mse',
    metrics=['mae']
)

# (3) 충분한 epoch으로 학습
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    verbose=2,
    validation_split=0.1
)

# (4) 테스트 평가
mse, mae = model.evaluate(x_test, y_test, verbose=0)
y_test_pred = model.predict(x_test)
test_r2     = r2_score(y_test, y_test_pred)

print("=== Test Performance ===")
print(f"Optimizer: {best_opt_cls.__name__}")
print(f"LR       : {best_lr}")
print(f"MSE      : {mse:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"R2       : {test_r2:.4f}")

# === Test Performance ===
# Optimizer: RMSprop
# LR       : 0.0001
# MSE      : 2784.5544
# MAE      : 37.5544
# R2       : 0.5825