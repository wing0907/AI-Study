import numpy as np
from sklearn.datasets import load_wine
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
datasets = load_wine()

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

# acc :  [0.96551724 1.         0.96428571 0.96428571 1.        ] 
# 평균 acc :  0.9788
# [1 0 1 2 2 0 1 1 1 0 1 1 0 0 2 1 2 1 1 2 0 2 1 2 1 0 2 1 0 0 0 2 0 1 0 2]
# [1 0 1 2 2 0 1 1 0 0 1 1 0 0 2 2 2 1 1 2 0 2 1 2 1 0 2 1 0 0 0 2 0 1 0 2]
# cross_val_predict ACC :  0.9444444444444444