# keras67_optimizer02_california 복붙
import sklearn as sk
#print(sk.__version__) #1.1.3
import tensorflow as tf
#print(tf.__version__) #2.9.3

from keras.models import Sequential
import numpy as np
import pandas as pd # 전처리 
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.metrics import r2_score
import time
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

#1. 데이터
dataset  = fetch_california_housing()
#dataset = pd.DataFrame(data=housing.data, columns=housing.feature_names)
#print(dataset.info())
#exit()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size= 0.2,random_state= 36) #6, 21, 36
#print(dataset.info())


#2. 모델 구성
model = Sequential()
model.add(Dense(400, input_dim = 8, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=20, verbose=1,
                   restore_best_weights=True,)

rlr = ReduceLROnPlateau(monitor='val_loss', mode='auto',
                        patience=10, verbose=1,
                        factor=0.5,
                        )

# 0.1 / 0.05 / 0.025 / 0.0125 / 0.00625 ##### 0.5
# 0.1 / 0.01 / 0.001 / 0.0001 / 0.00001 ##### 0.1
# 0.1 / 0.09 / 0.081 / 0.0729 / ...     ##### 0.9  factor 만큼 곱한다

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000005,
                 batch_size= 32,verbose=2,validation_split=0.1,
                 callbacks=[es,rlr])

end = time.time()

# 4. 평가 및 예측

# 4-1. 학습 시간 출력
print(f"걸린시간: {end - start:.2f}초")

# 4-2. 모델 평가 (Test set)
loss = model.evaluate(x_test, y_test, verbose=0)
print(f"Test MSE: {loss:.4f}")

# 4-3. 예측 및 R² 계산
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f"Test R2 Score: {r2:.4f}")

# Epoch 00022: early stopping
# 걸린시간: 39.91초
# Test MSE: 0.6948
# Test R2 Score: 0.4599