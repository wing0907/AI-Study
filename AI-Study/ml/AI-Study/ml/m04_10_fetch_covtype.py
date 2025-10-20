from sklearn.datasets import fetch_covtype
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression         # 이진분류 할 때 쓰는 놈 sigmoid 형태. 회귀냐 분류냐 유일하게 분류임
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
import ssl as ssl
from sklearn.model_selection import train_test_split
ssl._create_default_https_context = ssl._create_unverified_context

#  1.데이터
datasets = fetch_covtype(data_home='./fresh_data')
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))


y = y.reshape(-1, 1) 
encoder = OneHotEncoder(sparse=False) # 메트릭스형태를 받기때문에 n,1로 reshape하고 해야 한다.
y = encoder.fit_transform(y)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test =  train_test_split(
    x,y, test_size = 0.2, random_state=111, 
    stratify=y
)

print(np.min(x_train), np.max(x_train))   # 0.0 711.0
print(np.min(x_test), np.max(x_test))     # 0.0 711.0

print(x_train.shape, y_train.shape) # (464809, 54) (464809, 7)
print(x_test.shape, y_test.shape)   # (116203, 54) (116203, 7)


x_train = x_train.reshape(-1,54,1)
x_test = x_test.reshape(-1,54,1)


print(x_train.shape, y_train.shape) # (464809, 54, 1) (464809, 7)
print(x_test.shape, y_test.shape)   # (116203, 54, 1) (116203, 7)

## 최종 결과 예시 ==> ##
# LinearSVC : 0.7
# LogisticRegression : 0.8
# DecisionTreeClassifier : 0.9
# RandomForestClassifier : 1.0

# model_list = [LinearSVC, LogisticRegression, DecisionTreeClassifier, 
#               RandomForestClassifier,]

# 2. 모델 리스트 (이름과 클래스 쌍으로 저장)
model_list = [
    ('LinearSVC', LinearSVC),
    ('LogisticRegression', LogisticRegression),
    ('DecisionTreeClassifier', DecisionTreeClassifier),
    ('RandomForestClassifier', RandomForestClassifier)
]

for name, model_class in model_list:
    # 하이퍼파라미터는 여기서 추가
    if name == 'LinearSVC':
        model = model_class(max_iter=10000)
    elif name == 'LogisticRegression':
        model = model_class(max_iter=5000)
    else:
        model = model_class()

    model.fit(x, y)
    print(f'# {name} : {model.score(x, y):.4f}')


# LinearSVC : 0.9382
# LogisticRegression : 0.9944
# DecisionTreeClassifier : 1.0000
# RandomForestClassifier : 1.0000