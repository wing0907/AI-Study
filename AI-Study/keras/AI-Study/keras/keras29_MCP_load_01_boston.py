#copy from 27-3

import sklearn as sk
import pandas as pd
import numpy as np

# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]


from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_boston

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


# model = Sequential()

# model.add(Dense(32, input_dim = 13, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1))



path = './_save/keras28_mcp/01_boston/'
#1. 체크포인트로 확인
model = load_model(path + 'k28_0604_1135_0053-7.4029.hdf5')


model.summary()

model.compile(loss = 'mse', optimizer= 'adam')


# es = EarlyStopping(
#     monitor = 'val_loss',
#     mode = 'min',       
#     patience = 20,
#     restore_best_weights = True,
# )


# exit()

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,
#     filepath=filepath
# )


##list 형식으로 저장. 매 epoch 끝날 때마다 하나씩 들어가니, epoch 갯수만큼 
# hist = model.fit(x_train,y_train, epochs = 100000, batch_size =16,
#           verbose = 1,
#           validation_split = 0.2,
#           callbacks = [es, mcp],
          
#           )


# path = './_save/keras27_mcp/'
# model.save(path + 'keras27_3_save_model.h5')



#4 evaluate and predict

loss= model.evaluate(x_test,y_test)



result = model.predict(x_test)
r2 = r2_score(y_test, result)
print("loss:", loss)
print("r2 score", r2)
rmse = np.sqrt(mean_squared_error(y_test, result))
print("rmse", rmse)


# loss: 10.49239730834961
# r2 score 0.7818071561310002
# rmse 3.2391969089860826