#  08-1 copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y= np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape)  # (10,)
# print(y.shape)  # (10,)

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

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
          verbose=2
          )            
# 0 = 침묵. 빨리 넘기기
# 1 = default.
# 2 = 프로그래스바 삭제. 간결해짐
# 3 = 에포만 나옴. epoch 만 확인하고 싶으면 0,1,2 이외의 숫자 입력

# 4.  평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([11])

print('loss:', loss)
print('[11]의 예측값:', results)

# loss: 2.599335857667029e-09
# [11]의 예측값: [[11.000064]]             
             
# loss: 3.7462461477844045e-05
# [11]의 예측값: [[10.991597]]