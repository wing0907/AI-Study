import tensorflow as tf
print(tf.__version__)   # 2.9.3
import numpy as np
print(np.__version__)   # 1.21.1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000)

# 4. 평가, 예측
result = model.predict(np.array([4]))
print('4의 예측값 : ', result)





