# 배치를 적용한 거.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
X = np.array([1,2,3,4,5,6]) # 실질적으로 6 epochs가 적용된것
y = np.array([1,2,3,5,4,6]) # 1 batch = 6/6, 2 batch = 3/3, 3 batch = 2/2, 4 batch = 전체 사이즈의 50% 이상이면, 4개 훈련시키고 나머지 2개 훈련 시킨다 2/2

# 에포는 100으로 고정
# loss 기준 0.32 미만으로 만들 것.


# 2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(80))
model.add(Dense(630))
model.add(Dense(830))
model.add(Dense(50))
model.add(Dense(1))

epochs = 100
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=epochs, batch_size=3)    #덩어리 리스트를 쪼갠단위 batch / lower the batch, more efficient it is, but not always



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



# ############################### batch_size=3
# epochs: 100
# loss: 0.32381272315979004



