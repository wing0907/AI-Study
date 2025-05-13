# dacon, 데이터 파일 별도
# http://dacon.io/competitions/official/236068/overview/description

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,BatchNormalization, Dropout, Input
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping


# 1. 데이터
path = './_data/dacon/diabetes/'

                # [=] = b 를 a에 넣어줘 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # . = 현재위치, / = 하위폴더
print(train_csv)                  # [652 rows x 9 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)                   # [116 rows x 8 columns]

submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
print(submission_csv)             # [116 rows x 1 columns]


print(train_csv.shape)              # (652, 9)
print(test_csv.shape)               # (116, 8)
print(submission_csv.shape)         # (116, 1)

print(train_csv.info())
print(train_csv.describe())

print(test_csv.info())

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']


# x = x.replace(0, np.nan)

# train_csv = train_csv.replace(0, np.nan)
# train_csv = train_csv.fillna(train_csv.mean())
x = x.replace(0, np.nan)
x = x.fillna(x.mean())

print(train_csv.isna().sum()) 
print(train_csv.info()) 

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

print(test_csv.info())           

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = scaler.fit_transform(x)
test_csv = scaler.transform(test_csv)


print(x)                # [652 rows x 8 columns]
print(y.shape)          # (652,)


print(pd.DataFrame(y).value_counts())
# 1    228
# 0    424
print(pd.Series(y).value_counts())
# 1    228
# 0    424


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=55,                   # 33
    shuffle=True, 
)

print(x_train.shape, x_test.shape)      # (456, 8) (196, 8)
print(y_train.shape, y_test.shape)      # (456,) (196,)


# 2. 모델구성.
from tensorflow.keras.layers import Dropout, BatchNormalization

# model = Sequential()
# model.add(Dense(128, input_dim=8, activation='relu'))
# model.add(BatchNormalization())                             # Dropout 또는 BatchNormalization을 추가하면 학습 안정성 향상에 좋습니다.
# model.add(Dropout(0.3))

# model.add(Dense(64, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))  # 이진분류는 무조건 마지막 activation='sigmoid' 이다. node는 1개

input1 = Input(shape=(8,))
dense1 = Dense(128, activation='relu')(input1)
batch1 = BatchNormalization()(dense1)
drop1 = Dropout(0.3)(batch1)
dense2 = Dense(64, activation='relu')(drop1)
batch2 = BatchNormalization()(dense2)
drop2 = Dropout(0.3)(batch2)
dense3 = Dense(32, activation='relu')(drop2)
output1 = Dense(1, activation='sigmoid')(dense3)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()

