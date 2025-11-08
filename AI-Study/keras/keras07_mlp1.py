import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([[1,2,3,4,5],
              [6,7,8,9,10]])
y = np.array([1,2,3,4,5])
# x = np.array([[1,6], [2,7], [3,8], [4,9], [5,10]]) 의도했던 데이터 (5, 2)
x = np.transpose(x)

print(x.shape)  #   (2, 5) -> (5, 2)
print(y.shape)  #   (5,)


# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


# 4. 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[6, 11]])  # (1, 2)
print('loss:', loss)
print('[6,11]의 예측값:', results)

# loss: 7.048583721899759e-13
# [6,11]의 예측값: [[6.0000014]]



