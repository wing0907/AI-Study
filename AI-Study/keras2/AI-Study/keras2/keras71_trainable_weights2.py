'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

random.seed(333)
np.random.seed(333)
# tf.random.set_seed(333) 
print(tf.__version__)   # 2.7.4

# 1. 데이터
# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3,4,5])
x = np.array([1])
y = np.array([1])

# 2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

###### 동결 ######
# model.trainable = False             # 동결
model.trainable = True              # 동결 x    ★★★★★디폴트

print("="*100)
print(model.weights)
print("="*100)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=0)

# 4. 평가, 예측
y_pred = model.predict(x)
print(y_pred)

# x=1, y=1, 가중치 동결 후 훈련
# [[0.2949715]]
'''

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

random.seed(333)
np.random.seed(333)
tf.random.set_seed(333) 

x = np.array([1])
y = np.array([1])

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

print("="*50)
print("훈련 전 (Before training) weights & biases")
print("="*50)
print(model.weights)
print("="*50)

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=1000, verbose=0)

print("\n\n" + "="*50)
print("훈련 후 (After training) weights & biases")
print("="*50)
print(model.weights)
print("="*50)

y_pred = model.predict(x)
print(f"최종 예측값: {y_pred}")