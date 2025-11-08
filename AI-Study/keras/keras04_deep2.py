from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
X = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 에포는 100으로 고정
# loss 기준 0.32 미만으로 만들것.


# 2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=1))
model.add(Dense(88))
model.add(Dense(640))
model.add(Dense(33))
model.add(Dense(52))
model.add(Dense(1))

epochs = 100
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=epochs)

# 4. 평가, 예측
loss = model.evaluate(X, y)
print("###############################")
print('epochs:', epochs)
print('loss:', loss)
results = model.predict([6])
print('6의 예측값:', results)

# ###############################
# epochs: 100
# loss: 0.3239239454269409





