import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
               [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.transpose(x)

print(x.shape)  #   (3, 10) -> (10, 3)
print(y.shape)  #   (10,)


# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


# 4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[11, 2.0, -1]]) 
print('loss:', loss)
print('[11, 2.0, -1]의 예측값:', results)

# column이 많아져서 성능이 좋아짐
# loss: 5.51381152107816e-13
# [11, 2.0, -1]의 예측값: [[11.]]














