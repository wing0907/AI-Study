# https://www.kaggle.com/competitions/bike-sharing-demand

# 1. train_csv 에서 casual 과 registered 를 y 로 잡느다.
# 2. 훈련해서, test_csv의 casual 과 registered 를 예측(predict) 한다.
# 3. 예측한 casual 과 registered 를 test_csv 에 column 으로 넣는다.
#   (N, 8) -> (N, 10) test.csv 파일로 new_test.csv 파일을 만든다.

import numpy as np
import pandas as pd
import sklearn as sk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 1. 데이터
path = './_data/kaggle/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

print(train_csv)                # [10886 rows x 11 columns]
print(test_csv)                 # [6493 rows x 8 columns]
print(train_csv.shape)          # (10886, 11)
print(test_csv.shape)           # (6493, 8)
print(submission_csv.shape)     # (6493, 2)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)                        # [10886 rows x 8 columns]
y = train_csv[['casual', 'registered']]
print(y.shape)                  #  (10886, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=715,
    )

# 2. 모델구성
model = Sequential()
model.add(Dense(300, input_dim=8, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(700, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(700, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(2))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

# y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)

test2_csv = test_csv        # 원래는 .copy() 를 사용해야함.

test2_csv[['casual', 'registered']] = y_submit   # 사본을 만들어서 원본을 유지하기
print(test2_csv)

test2_csv.to_csv(path + 'new_test.csv', )       #, index=False

print('test_csv 타입 : ', type(test_csv))   # <class 'pandas.core.frame.DataFrame'>
print('y_submit 타입 : ', type(y_submit))   # <class 'numpy.ndarray'>

# exit()

r2 = r2_score(y_test, results)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, results)

# y_submit = model.predict(test_csv)
print('[loss]:', loss)
print('[rmse]:', rmse)
print('[r2]:', r2)


# [loss]: 9370.6396484375
# [rmse]: 96.80206539369084
# [r2]: 0.4266243237448437