import numpy as np
from sklearn.datasets import load_iris
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
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# print(train_csv)
# print(train_csv.head())           # 앞부분 5개 디폴트
# print(train_csv.tail())           # 뒷부분 5개
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


from sklearn.preprocessing import StandardScaler

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


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8,
    stratify=y
)



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

# acc :  [0.71290616 0.78929031 0.78897936 0.78848703 0.79284226] 
# 평균 acc :  0.7745
# id
# 110916    1
# 81797     0
# 155703    0
# 44621     0
# 11314     0
#          ..
# 160424    0
# 23945     0
# 27297     0
# 67177     1
# 120419    0
# Name: Exited, Length: 33007, dtype: int64
# [0 0 1 ... 0 0 0]
# cross_val_predict ACC :  0.7548701790529282