import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.utils import class_weight
import time


# 1. 데이터
path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)



x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape, y.shape)     # (61878, 93) (61878,)


for scaler in [StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()]:
    x_scaled = scaler.fit_transform(x)
    # → 모델 학습 & 검증


# 1. 스케일링
scaler = RobustScaler()
x_scaled = scaler.fit_transform(x)

# 2. 레이블 유지한 채 train/test split
y_label = train_csv['target'].values
x_train, x_test, y_train_label, y_test_label = train_test_split(
    x_scaled, y_label, test_size=0.2, random_state=55
)



# 4. One-Hot Encoding
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train_label.reshape(-1, 1))
y_test = ohe.transform(y_test_label.reshape(-1, 1))

class_names = ohe.categories_[0]  # array of strings
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=class_names,
    y=y_train_label
)

class_weights = dict(enumerate(weights))


print(x_train.shape, y_train.shape) # (49502, 93) (49502, 9)
print(x_test.shape, y_test.shape)   # (12376, 93) (12376, 9)


x_train = x_train.reshape(-1,93,1)
x_test = x_test.reshape(-1,93,1)


print(x_train.shape, y_train.shape) # (49502, 93, 1) (49502, 9)
print(x_test.shape, y_test.shape)   # (12376, 93, 1) (12376, 9)


model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2,
                 padding='same', activation='relu', input_shape=(93,1)))      
model.add(Conv1D(128, 2))                              
model.add(Flatten())                           
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128,  activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=64, input_shape=(128,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=9, activation='softmax'))
model.summary()


model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
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
filepath = "".join([path, 'k41_13', date, '_', filename])

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
print('loss : ', loss[0])
print('acc : ', loss[1])
y_pred = model.predict(x_test)

y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)


acc = accuracy_score(y_test, y_pred)

print('acc:' , round(acc, 4))
print('걸린시간:', end - start, '초')

# loss :  0.5463557243347168
# acc :  0.7859566807746887
# acc: 0.786
# 걸린시간: 22.971617937088013 초

# loss :  2.007565975189209
# acc :  0.31738850474357605
# acc: 0.3174
# 걸린시간: 75.26668977737427 초


# loss :  0.5343830585479736
# acc :  0.7857950925827026
# acc: 0.7858
# 걸린시간: 15.233550786972046 초