#  train.csv 와 new_test.csv로 count 예측

import numpy as np
import pandas as pd
# import sklearn as sk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# print(sk.__version__)


# exit()
# 1. 데이터
path = './_data/kaggle/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)           # 가독성을 위해 new_test_csv 로 기입
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

print(train_csv)                # [10886 rows x 11 columns]
print(new_test_csv)                 # [6493 rows x 10 columns]
print(train_csv.shape)          # (10886, 11)
print(new_test_csv.shape)           # (6493, 10)
print(submission_csv.shape)     # (6493, 2)

x = train_csv.drop(['count'], axis=1)       # [10886 rows x 10 columns]
y = train_csv['count']                      # (10886,)
print(x)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=361,
    )


# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')


import time
start = time.time()
hist = model.fit(x_train,y_train, epochs = 100, batch_size =32,
          verbose = 1,
          validation_split = 0.2,       
          )
end = time.time()


import tensorflow as tf
gpus =tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')



# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])
y_predict = model.predict(x_test)

r2 = r2_score(y_test, results)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, results)

y_submit = model.predict(new_test_csv)

print(submission_csv)
submission_csv['count'] = y_submit
print(submission_csv)



print('[loss]:', loss)
print('[rmse]:', rmse)
print('[r2]:', r2)
print("걸린시간 :", end - start, '초')

# cpu 사용결과
# [loss]: 0.009113866835832596
# [rmse]: 0.09546664825805368
# [r2]: 0.9999997259031712
# 걸린시간 : 25.09980845451355 초

# gpu 사용결과
# [loss]: 0.004376033321022987
# [rmse]: 0.06615159062674382
# [r2]: 0.999999868392317
# 걸린시간 : 79.46326446533203 초