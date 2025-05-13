# Keras67_gpu_test02_california 복붙
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
best_optim = ''
best_lr = 0
best_r2 = 0
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop
optim = [Adam, Adagrad,SGD,RMSprop]
lr = [0.1,0.01,0.05,0.001,0.0001]

for op in optim:
    for i in lr:
        
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
        model.compile(
            optimizer=op(learning_rate=i),  # ✅ 객체를 그대로 넘김
            loss='mse',
            metrics=['mae']
        )

        start = time.time()
        hist = model.fit(x_train, y_train,epochs= 1, batch_size= 32,verbose=2,validation_split=0.1)#,class_weight=class_weights,)
        end = time.time()
        
        x_pred = model.predict(x_test)
        if not np.isnan(x_pred).any() and not np.isnan(y_test).any():
            r2 = r2_score(y_test, x_pred)
            print(r2)
            if r2 > best_r2:
                best_r2 = r2
                best_optim = f'{op}'
                best_lr = i
        else:
            print("NaN detected in prediction or test labels!")
        

print('best_r2:',r2,'best_optim:',best_optim,"best_lr:",best_lr)
print("걸린시간 :",end-start)
# GPU 있다
# 걸린시간 : 138.6830358505249
# GPU 없다
# 걸린시간 : 70.0864520072937