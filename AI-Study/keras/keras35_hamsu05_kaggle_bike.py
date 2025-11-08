#  train.csv 와 new_test.csv로 count 예측

import numpy as np
import pandas as pd
# import sklearn as sk
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# print(sk.__version__)


# exit()
# 1. 데이터
path = './_data/kaggle/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)           # 가독성을 위해 new_test_csv 로 기입
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

print(train_csv)                # [10886 rows x 11 columns]
print(new_test_csv)                 # [6493 rows x 10 columns]
print(train_csv.shape)          # (10886, 11)
print(new_test_csv.shape)           # (6493, 10)
print(submission_csv.shape)     # (6493, 2)

x = train_csv.drop(['count'], axis=1)       # [10886 rows x 10 columns]
y = train_csv['count']                      # (10886,)
print(x)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=361,
    )


# 2. 모델구성
# model = Sequential()
# model.add(Dense(100, input_dim=10, activation='relu'))
# model.add(Dense(400, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1))

input1 = Input(shape=(10,))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(400, activation='relu')(dense1)
dense3 = Dense(100, activation='relu')(dense2)
output1 = Dense(1)(dense3)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()