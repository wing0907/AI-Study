#  08-1 copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y= np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape)  # (10,)
# print(y.shape)  # (10,)

x_train = np.array([1,2,3,4,5,6,])
y_train = np.array([1,2,3,4,5,6,])

x_val = np.array([7,8,])        # train set은 validation과 train 분류한다. 데이터가 아깝지만 나중에 다 활용함
y_val = np.array([7,8,])        # 성능 향상에는 도움이 되지 않는다.

x_test = np.array([9,10])       # test data는 훈련에 사용하지 않는다.
y_test = np.array([9,10])

# exit()

# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=3,
          validation_data=(x_val, y_val)
          )             # 1 epoch 당 validation 1 회

# 4.  평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([11])

print('loss:', loss)
print('[11]의 예측값:', results)

# loss: 2.599335857667029e-09
# [11]의 예측값: [[11.000064]]             
             
# loss: 3.7462461477844045e-05
# [11]의 예측값: [[10.991597]]