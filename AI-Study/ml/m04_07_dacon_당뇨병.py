from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression         # 이진분류 할 때 쓰는 놈 sigmoid 형태. 회귀냐 분류냐 유일하게 분류임
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


#  1.데이터
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']
print(x)
print(y)
print(x.shape, y.shape)

# 3. Replace 0s with NaN (only in specific columns)
zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_not_allowed] = x[zero_not_allowed].replace(0, np.nan)

# 4. Fill NaNs with mean
x = x.fillna(x.mean())

# 5. Scale Data
scaler = RobustScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_csv)

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
        model = model_class(max_iter=1000)
    else:
        model = model_class()

    model.fit(x_scaled, y)
    print(f'# {name} : {model.score(x_scaled, y):.4f}')


# LinearSVC : 0.7699
# LogisticRegression : 0.7699
# DecisionTreeClassifier : 1.0000
# RandomForestClassifier : 1.0000