
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression         # 이진분류 할 때 쓰는 놈 sigmoid 형태. 회귀냐 분류냐 유일하게 분류임
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


#  1.데이터
digits = load_digits()
x, y = digits.data, digits.target
print(x)
print(y)
print(x.shape, y.shape)


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


# LinearSVC : 0.9950
# LogisticRegression : 1.0000
# DecisionTreeClassifier : 1.0000
# RandomForestClassifier : 1.0000