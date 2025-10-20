from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]])
            
#   [실습]            
#   loss와 [[10, 31, 211]]을 예측하시오.

x = x.T
y = y.T
print(x.shape)  # (10, 3)
print(y.shape)  # (10, 3)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=3))
model.add(Dense(400))
model.add(Dense(800))
model.add(Dense(200))
model.add(Dense(60))
model.add(Dense(3))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=150, batch_size=3)

# 4. 평가, 예측
print('##############################')
loss = model.evaluate(x, y)
results = model.predict([[10, 31, 211]])
print('loss:', loss)
print('[10, 31, 211]의 예측값:', results)

# loss: 0.03861428424715996
# [10, 31, 211]의 예측값: [[11.087203    0.03299673 -0.33528885]]

# loss: 0.021043632179498672
# [10, 31, 211]의 예측값: [[11.052917   0.4098669 -1.1861259]]