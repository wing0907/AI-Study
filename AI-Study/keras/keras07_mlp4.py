import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 1. 데이터
x = np.array([range(10)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1,],
              [9,8,7,6,5,4,3,2,1,0]])
print(x)
print(y.shape) # (3, 10)

x = x.T
y = y.T

# 2. 모델 구성
model = Sequential()
model.add(Dense(200, input_dim=1))
model.add(Dense(400))
model.add(Dense(800))
model.add(Dense(200))
model.add(Dense(60))
model.add(Dense(3))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=3)

# 4. 평가, 예측
print('################')
loss = model.evaluate(x, y)
results = model.predict([10])
print('loss:', loss)
print('[10]의 예측값:', results)

# loss: 0.0010715191019698977
# [10]의 예측값: [[10.995821   -0.07494533 -1.0699139 ]]





