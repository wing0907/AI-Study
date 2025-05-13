# https://www.kaggle.com/competitions/playground-series-s4e1/submissions

# copy from 21_2 (keras)

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
import time
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC
import random
from sklearn.metrics import accuracy_score, f1_score, r2_score


seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)



x = train_csv.drop(['target'], axis=1)
y = train_csv['target']


# 2. 라벨 인코딩
le = LabelEncoder()
y_enc = le.fit_transform(y)   # now 0~8



x_train, x_test, y_train, y_test = train_test_split(
    x, y_enc, train_size=0.8, random_state=seed,
    stratify=y_enc
)

print(train_csv.info())


categorical_features = list(range(1, 93)) 
smotenc = SMOTENC(random_state=337, categorical_features=categorical_features)

x_res, y_res = smotenc.fit_resample(x, y)

print(x_res)



scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)
print("================", model.__class__.__name__, "================")
print('acc : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy_score :', acc)
f1 = f1_score(y_test, y_pred, average='macro')
print('f1_score :', f1)


# ================ KNeighborsClassifier ================
# acc :  0.7720588235294118
# accuracy_score : 0.7720588235294118
# f1_score : 0.715966715225559