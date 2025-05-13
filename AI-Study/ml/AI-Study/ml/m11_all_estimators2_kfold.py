import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__) # 0.24.2      --> 1.6.1
from sklearn.utils import class_weight

path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

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
    x_scaled, y, train_size=0.8, random_state=588
)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

print(x_train.shape, y_train.shape) # (132027, 10) (132027,)
print(x_test.shape, y_test.shape)   # (33007, 10) (33007,)
x_train = x_train.reshape(-1,5,2)
x_test = x_test.reshape(-1,5,2)
print(x_train.shape, y_train.shape) # (132027, 5, 2) (132027,)
print(x_test.shape, y_test.shape)   # (33007, 5, 2) (33007,)


n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)
# 2. 모델구성
# model = RandomForestRegressor()
# allAlgorithms = all_estimators(type_filter='classifier')

# print('allAlgorithms : ', allAlgorithms)
# print('모델의 갯수 : ', len(allAlgorithms)) # 모델의 갯수 :  55
# print(type(allAlgorithms))                 # <class 'list'>


# max_name = ""
# max_score = 0

# for (name, algorithms) in allAlgorithms:
#     try:
#         model = algorithms()
#         model.fit(x_train, y_train)
#         results = model.score(x_test, y_test)
#         print(name, '의 정답률 :', results)

#         if results > max_score:
#             max_score = results
#             max_name = name
#     except:
#         print(name, "은(는) 에러뜬 분!!!")

# print("============================================")
# print("최고모델 :", max_name, max_score)
# print("============================================")

# 최고모델 : DummyClassifier 0.7882267397824704

# 2. 모델
model = DummyClassifier()

# 3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc : ', scores ,'\n평균 acc : ', round(np.mean(scores), 4))


y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)


acc = accuracy_score(y_test, y_pred)
print('cross_val_predict ACC : ', acc)

# acc :  [0.86779544 0.86865601 0.86753727 0.86815043 0.8617915 ] 
# 평균 acc :  0.8668
# cross_val_predict ACC :  0.842293228229908