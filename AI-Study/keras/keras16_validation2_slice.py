import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#  1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

#  [실습] 리스트의 슬라이싱으로 10:4:3 으로 나눈다.

x_train =  x[:10]
y_train = y[:10]

x_val =  x[10:14]
y_val = y[10:14]

x_test = x[14:]
y_test = y[14:]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=15,
)

#  2. 모델구성
model = Sequential()
model.add(Dense(20, input_dim=1, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))



#  3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1,
          validation_data=(x_val, y_val))


#  4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

results = model.predict([17])
print('[17]의 예측값 : ', results)
