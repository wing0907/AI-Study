import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array(range(10))    # 끝 -1 , 직전까지
print(x)        #   [0 1 2 3 4 5 6 7 8 9]
print(x.shape)  # (10,) 스칼라 10개짜리 벡터

x = np.array(range(1, 10)) # 1부터 9까지
print(x)        #   [1 2 3 4 5 6 7 8 9]

x = np.array(range(1, 11)) # 1부터 10
print(x)        #   [ 1  2  3  4  5  6  7  8  9 10]

x = np.array([range(10), range(21, 31), range(201, 211)])
print(x)
print(x.shape)  # (3, 10)

x = x.T
print(x)
print(x.shape)  # (10, 3)

y = np.array([1,2,3,4,5,6,7,8,9,10])

# [실습]
#  [10, 31, 211] 예측

#2. 모델 구성
model = Sequential()
model.add(Dense(200, input_dim=3))
model.add(Dense(200))
model.add(Dense(400))
model.add(Dense(800))
model.add(Dense(400))
model.add(Dense(40))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가 예측
loss = model.evaluate(x, y)
results = model.predict([[10, 31, 211]])    # (1,3)
print('loss:', loss)
print('[10, 31, 211]의 예측값:', results)

# loss: 1.7003287666739197e-11
# [10, 31, 211]의 예측값: [[10.999987]]

