import numpy as np
import pandas as pd               # 전처리
print(np.__version__)             # 1.23.0
print(pd.__version__)             # 2.2.3

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,  BatchNormalization, Dropout, Input, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

#1. 데이터
path = './_data/dacon/따릉이/'

                # [=] = b 를 a에 넣어줘 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # . = 현재위치, / = 하위폴더
print(train_csv)                  # [1459 rows x 11 columns] -> [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)                   # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission_csv)             # [715 rows x 1 columns]

print(train_csv.shape)            #(1459, 10)
print(test_csv.shape)             #(715, 9)
print(submission_csv.shape)       #(715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')           # Nan = 결측치              이상치 ex. 서울의 온도 41도

print(train_csv.info())           # 결측치 확인

print(train_csv.describe())

########################   결측치 처리 1. 삭제   ######################
# print(train_csv.isnull().sum()) # 결측치의 개수 출력
print(train_csv.isna().sum())     # 위 함수와 똑같음

train_csv = train_csv.dropna()  #결측치 처리를 삭제하고 남은 값을 반환해 줌
print(train_csv.isna().sum()) 
print(train_csv.info())         # 결측치 확인
print(train_csv)                # [1328 rows x 10 columns]

########################   결측치 처리 2. 평균값 넣기   ######################
# train_csv = train_csv.fillna(train_csv.mean())
# print(train_csv.isna().sum()) 
# print(train_csv.info()) 

########################   테스트 데이터의 결측치 확인   ######################
print(test_csv.info())            # test 데이터에 결측치가 있으면 절대 삭제하지 말 것!
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())            # 715 non-null

#====================== x 와 y 데이터를 나눠준다 =========================#
x = train_csv.drop(['count'], axis=1)    # pandas data framework 에서 행이나 열을 삭제할 수 있다
                #  count라는 axis=1 열 삭제, 참고로 행 삭제는 axis=0
print(x)                                 # [1459 rows x 9 columns]
y = train_csv['count']                   # count 컬럼만 빼서 y 에 넣겠다
print(y.shape)                           #(1459,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=123
    )            

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape) # (1195, 9) (1195,)
print(x_test.shape, y_test.shape)   # (133, 9) (133,)

x_train = x_train.reshape(-1,9,1,1)
x_test = x_test.reshape(-1,9,1,1)
print(x_train.shape, y_train.shape) # (1195, 9, 1, 1) (1195,)
print(x_test.shape, y_test.shape)   # (133, 9, 1, 1) (133,)



model = Sequential()
model.add(Conv2D(128, (1,1), strides=1, padding='same', input_shape=(9,1,1)))                                    
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
model.add(Dense(units=1, activation='linear'))
model.summary()


model.compile(loss= 'mse', optimizer='adam')
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
filepath = "".join([path, 'k41_4', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=512,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp],)
end = time.time()

 
 
loss= model.evaluate(x_test, y_test, verbose=1)


y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('r2:' , round(r2, 4))
print('걸린시간:', end - start, '초')

# r2: 0.3304
# 걸린시간: 31.016581773757935 초