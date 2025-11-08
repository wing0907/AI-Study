

from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression         # 이진분류 할 때 쓰는 놈 sigmoid 형태. 회귀냐 분류냐 유일하게 분류임
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import OneHotEncoder


#  1.데이터
path = 'C:\Study25\_data\kaggle\santander\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)


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