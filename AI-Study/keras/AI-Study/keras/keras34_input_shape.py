# copy from 33_01
import sklearn as sk
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout #(dropout : 과적합방지에 최고)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_boston
import time

dataset = load_boston()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target


print(x.shape, y.shape)

# (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 190,
    
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))


model = Sequential()
# model.add(Dense(32, input_dim = 13, activation='relu'))
model.add(Dense(32, input_shape=(13,), activation='relu'))
model.add(Dropout(0.3))     # 상위 layer의 30% 가 빠지는 것
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))     
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))     
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))     
model.add(Dense(1, activation='linear'))



model.compile(loss = 'mse', optimizer= 'adam')


##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
start = time.time()
hist = model.fit(x_train,y_train, epochs = 100, batch_size =32,
          verbose = 1,
          validation_split = 0.2,       
          )
end = time.time()


import tensorflow as tf
gpus =tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')



#4 evaluate and predict

loss= model.evaluate(x_test,y_test)



result = model.predict(x_test)
r2 = r2_score(y_test, result)
print("loss:", loss)
print("r2 score", r2)
rmse = np.sqrt(mean_squared_error(y_test, result))
print("rmse", rmse)
print("걸린시간 :", end - start, '초')


# cpu 사용결과
# loss: 9.96886157989502
# r2 score 0.7926942566122614
# rmse 3.1573502757840926
# 걸린시간 : 2.94284987449646 초


# dropout
# loss: 12.223445892333984
# r2 score 0.7458094268505973
# rmse 3.496204424791532
# 걸린시간 : 3.161036252975464 초


# gpu 사용결과
# loss: 9.839776992797852
# r2 score 0.7953785929581422
# rmse 3.136841906526097
# 걸린시간 : 13.95410442352295

# dropout
# loss: 15.655436515808105
# r2 score 0.6744400665437791
# rmse 3.956694993454291
# 걸린시간 : 6.7868242263793945 초