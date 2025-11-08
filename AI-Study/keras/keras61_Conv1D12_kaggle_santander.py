
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
import time

#  1. data
path = 'C:\Study25\_data\kaggle\santander\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
print(train_csv)        # [200000 rows x 201 columns]



x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 190,)



scaler = MinMaxScaler()
scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

print(x_train.shape, y_train.shape) # (160000, 200) (160000,)
print(x_test.shape, y_test.shape)   # (40000, 200) (40000,)


x_train = x_train.reshape(-1,200,1)
x_test = x_test.reshape(-1,200,1)


print(x_train.shape, y_train.shape) # (160000, 200, 1) (160000,)
print(x_test.shape, y_test.shape)   # (40000, 200, 1) (40000,)

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2,
                 padding='same', activation='relu', input_shape=(200,1)))     
model.add(Flatten())                               
model.add(Dense(units=512, activation='relu'))                            
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=64, input_shape=(128,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()


model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=3, verbose=1,
                   restore_best_weights=True,
                   )

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")     # string 문자열

path = './_save/keras41/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k41_12', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=5, batch_size=512,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp],)
end = time.time()

 
 
loss= model.evaluate(x_test, y_test, verbose=1)


y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('r2:' , round(r2, 4))
print('걸린시간:', end - start, '초')

# r2: 0.1426
# 걸린시간: 104.05239081382751 초

# r2: -0.0017
# 걸린시간: 879.7224698066711 초

# r2: 0.2953
# 걸린시간: 34.99814748764038 초