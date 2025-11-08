from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y= np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape)  # (10,)
# print(y.shape)  # (10,)

# [실습] 넘파이 리스트의 슬라이싱 (google it)

x_test = x[7:10]
print(x_test)
y_test = y[7:]      # 시작과 끝은 명시 안해줘도 됨
print(y_test)

x_train = x[0:7]    # 7-1, 시작은 0부터
print(x_train)
y_train = y[:7]     # 0은 생략할 수 있다
print(y_train)

print(x_train.shape, y_train.shape)     # (7,)
print(x_test.shape, y_test.shape)     # (3,)



# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

# exit()

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=3)

# 4.  평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([11])

print('loss:', loss)
print('[11]의 예측값:', results)

# loss: 2.599335857667029e-09
# [11]의 예측값: [[11.000064]]             
             
# loss: 3.7462461477844045e-05
# [11]의 예측값: [[10.991597]]

