from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1]])

x = x.T      #  (3, 10)
y = y.T     # (2, 10)
print(x.shape)   #  (10, 3)
print(y.shape)  #   (10, 2)    

#[실습]
# loss와 [[10, 31, 211]]을 예측하시오.


#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=3))
model.add(Dense(400))
model.add(Dense(800))
model.add(Dense(200))
model.add(Dense(60))
model.add(Dense(2))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=3)

# 4. 평가, 예측
print('########################')
loss = model.evaluate(x, y)
results = model.predict([[10, 31, 211], [11, 32, 212]])  # (2,3)
print('loss:', loss)
print('[11, 0]의 예측값:', results)

# loss: 0.03008180856704712
# [10, 31, 211]의 예측값: [[10.614176  -0.2224974]
#  [11.54034   -1.2646871]]

