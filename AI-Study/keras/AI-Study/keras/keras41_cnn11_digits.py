from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dropout, MaxPooling2D, Flatten, Dense, BatchNormalization
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

# 1. 데이터 로드
digits = load_digits()
x, y = digits.data, digits.target

# x = np.min(x), np.max(x)
# print(x)  # (0.0, 16.0)


# 2. 학습/테스트 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=digits.target, random_state=555
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape) # (1437, 64) (1437,)
print(x_test.shape, y_test.shape)   # (360, 64) (360,)


x_train = x_train.reshape(-1,64,1,1)
x_test = x_test.reshape(-1,64,1,1)


print(x_train.shape, y_train.shape) # (1437, 64, 1, 1) (1437,)
print(x_test.shape, y_test.shape)   # (360, 64, 1, 1) (360,)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape) # (1437, 10) (360, 10)


model = Sequential()
model.add(Conv2D(128, (1,1), strides=1, padding='same', input_shape=(64,1,1)))                                    
model.add(Conv2D(filters=512, kernel_size=(1,1)))                            
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(128, (1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=64, input_shape=(128,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))
model.summary()


model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=50, verbose=1,
                   restore_best_weights=True,
                   )

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")     # string 문자열

path = './_save/keras41/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k41_11', date, '_', filename])

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

# r2: 0.4308
# 걸린시간: 5.169618368148804 초