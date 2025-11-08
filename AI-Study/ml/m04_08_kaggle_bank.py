
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression         # 이진분류 할 때 쓰는 놈 sigmoid 형태. 회귀냐 분류냐 유일하게 분류임
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


#  1.데이터
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

print(train_csv.head(10))           # 앞부분 10개          

print(train_csv.isna().sum())       # train data의 결측치 갯수 확인  -> 없음
print(test_csv.isna().sum())        # test data의 결측치 갯수 확인   -> 없음

print(train_csv.columns)
# Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
    #    'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
    #    'EstimatedSalary', 'Exited']

#  문자 데이터 수치화!!!
from sklearn.preprocessing import LabelEncoder

le_geo = LabelEncoder()
le_gender = LabelEncoder()

train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])
test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

print(train_csv['Geography'])
print(train_csv['Geography'].value_counts())         # 잘 나왔는지 확인하기. pandas는 value_counts() 사용
# 0    94215
# 2    36213
# 1    34606
print(train_csv['Gender'].value_counts())
# 1    93150
# 0    71884

train_csv = train_csv.drop(['CustomerId', 'Surname',], axis=1)  # 2개 이상은 리스트
test_csv = test_csv.drop(['CustomerId', 'Surname', ], axis=1)


x = train_csv.drop(['Exited'], axis=1)
print(x.shape)      # (165034, 10)
y = train_csv['Exited']
print(y.shape)      # (165034,)

# 1. 컬럼 분리
x_other = x.drop(['EstimatedSalary'], axis=1)
x_salary = x[['EstimatedSalary']]

# 2. 각각 스케일링
scaler_other = StandardScaler()
scaler_salary = StandardScaler()

x_other_scaled = scaler_other.fit_transform(x_other)
x_salary_scaled = scaler_salary.fit_transform(x_salary)

# 3. 합치기
x_scaled = np.concatenate([x_other_scaled, x_salary_scaled], axis=1)

# 4. test set도 동일하게 처리
test_other = test_csv.drop(['EstimatedSalary'], axis=1)
test_salary = test_csv[['EstimatedSalary']]

test_other_scaled = scaler_other.transform(test_other)
test_salary_scaled = scaler_salary.transform(test_salary)

test_scaled = np.concatenate([test_other_scaled, test_salary_scaled], axis=1)



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


# LinearSVC : 0.8222
# LogisticRegression : 0.8261
# DecisionTreeClassifier : 0.9997
# RandomForestClassifier : 0.9997