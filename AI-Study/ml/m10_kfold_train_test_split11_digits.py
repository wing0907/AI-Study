import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
datasets = load_digits()

x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8,
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

# 2. 모델
model = MLPClassifier()

# 3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc : ', scores ,'\n평균 acc : ', round(np.mean(scores), 4))


y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_test)
print(y_pred)


acc = accuracy_score(y_test, y_pred)
print('cross_val_predict ACC : ', acc)

# acc :  [0.98611111 0.97916667 0.96515679 0.97560976 0.96864111] 
# 평균 acc :  0.9749
# [5 9 9 6 1 6 6 9 8 7 4 2 1 4 3 1 4 7 0 1 4 8 2 7 0 3 8 0 4 9 8 3 2 3 8 3 1
#  4 2 5 4 5 2 6 1 3 1 7 6 2 7 4 6 2 9 5 1 5 5 3 9 9 4 8 2 9 6 1 0 2 6 3 3 0
#  9 1 4 3 3 9 8 7 0 0 1 3 2 8 0 3 9 5 4 0 7 1 5 7 2 9 8 8 5 3 0 3 1 9 4 5 0
#  4 1 6 8 8 7 9 2 4 6 0 7 2 6 2 7 5 3 6 5 1 0 7 1 0 6 4 8 5 3 4 0 8 9 6 5 6
#  2 1 2 3 3 6 1 5 7 9 8 7 9 1 0 2 7 7 7 0 8 6 0 0 4 6 9 6 0 6 4 2 4 1 3 0 2
#  8 6 9 1 0 2 7 4 4 4 7 4 5 9 4 7 2 6 9 5 8 9 0 8 8 1 9 2 5 5 1 7 0 7 7 0 4
#  2 3 7 9 3 1 6 0 5 6 0 2 2 6 6 3 4 3 8 3 9 9 6 1 6 5 2 4 7 9 5 5 4 5 0 7 3
#  0 3 0 1 8 7 4 3 0 8 1 7 6 9 9 6 3 2 9 5 7 1 3 1 0 0 5 5 3 8 8 8 7 1 2 9 8
#  4 1 4 7 0 2 9 1 4 3 1 5 1 4 9 2 2 3 8 7 6 5 2 1 5 4 6 5 5 6 6 5 8 0 2 1 8
#  7 8 3 8 3 2 9 8 7 5 2 8 0 4 6 7 5 7 5 9 8 9 3 4 3 5 6]
# [5 9 9 6 1 6 6 9 8 7 4 2 1 4 3 1 4 7 0 1 4 1 2 7 0 3 8 0 4 9 1 3 2 3 8 3 1
#  4 2 5 4 5 2 6 1 3 1 7 6 2 7 4 6 2 9 5 1 5 5 3 9 9 4 8 2 9 6 1 0 2 6 3 3 0
#  9 1 4 3 3 9 8 7 0 0 1 3 2 8 0 9 9 5 4 0 7 1 5 7 2 9 8 8 5 3 0 3 1 9 4 5 0
#  4 1 6 8 8 7 9 2 4 6 0 7 2 6 2 7 5 3 6 5 1 0 7 1 0 6 4 8 5 3 4 0 8 9 1 5 6
#  2 1 2 3 3 6 1 5 7 9 8 7 8 1 0 2 7 7 7 0 8 6 0 0 4 6 9 6 0 6 4 7 4 1 3 0 2
#  8 6 9 1 0 2 7 4 4 4 7 4 5 9 4 7 2 6 9 5 8 9 0 2 2 1 8 2 5 5 1 7 0 7 7 0 4
#  2 3 5 9 7 1 6 0 5 6 0 2 2 6 6 3 4 3 8 3 9 9 6 1 6 5 2 4 7 9 5 5 4 5 0 7 3
#  0 2 0 8 8 7 4 3 0 8 1 7 6 9 9 6 3 2 9 5 7 1 3 9 0 0 5 5 3 8 8 8 7 1 2 9 8
#  4 9 4 7 0 2 9 1 4 3 1 5 1 4 9 2 1 3 8 7 6 5 2 1 5 4 6 5 5 6 6 5 8 0 2 1 8
#  7 8 3 8 3 2 9 9 7 5 2 8 0 4 6 7 5 7 5 9 8 7 3 4 3 5 6]
# cross_val_predict ACC :  0.95