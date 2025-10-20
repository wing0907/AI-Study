import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

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

print(x.shape, y.shape)                     # (165034, 10) (165034,)
print(np.unique(y, return_counts=True))     # (array([0, 1], dtype=int64), array([130113,  34921], dtype=int64))
print(pd.value_counts(y))
# 0    130113
# 1     34921
print(y)


# exit()

### 데이터 삭제. 라벨이 2인놈을 10개만 남기고 다 지워라!!! 리스트의 슬라이싱으로 지우기
# x = x[:-40]
# y = y[:-40]
# print(y)

# print(np.unique(y, return_counts=True))     # (array([0, 1, 2]), array([59, 71,  8], dtype=int64))

# numpy 배열 y를 기반으로 원하는 개수만큼 슬라이싱
# idx_1 = np.where(y == 1)[0][:71]   # 클래스 1 → 71개
# idx_0 = np.where(y == 0)[0][:59]   # 클래스 0 → 59개
# idx_2 = np.where(y == 2)[0][:8]    # 클래스 2 → 8개

# # # 인덱스 합치기
# selected_idx = np.concatenate([idx_1, idx_0, idx_2])

# # # X, y 슬라이싱
# x_sub = x[selected_idx]
# y_sub = y[selected_idx]

# print(x_sub.shape, y_sub.shape)  # (138, 13) (138,)
# print(np.unique(y_sub, return_counts=True))  # 확인용


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.75, shuffle=True,
    stratify=y
)

print(np.unique(y_train, return_counts=True))  # 확인용
#  (array([0, 1, 2]), array([44, 53,  6], dtype=int64))


############## SMOTE 적용 ###############
# pip install imblearn
from imblearn.over_sampling import SMOTE, RandomOverSampler
import sklearn as sk
print('sklearn version : ', sk.__version__) # sklearn version :  1.6.1
import imblearn
print('imblearn version :', imblearn.__version__) # imblearn version : 0.12.4

ros = RandomOverSampler(random_state=seed,
              # k_neighbors=5, # default
              sampling_strategy='auto', # default. 머리쓰기 싫을땐 디폴트가 아름답다.
            #   sampling_strategy=0.75, # 최대값의 75% 지점
              # sampling_strategy={0:500, 1:500}, # 이렇게 하면 1.0 나옴. 아름다운 데이터일 경우.
              # n_jobs=-1, # 0.13에서는 삭제됨. 기냥 포함됨          
              )

x_train, y_train = ros.fit_resample(x_train, y_train) # smote.fit_resample 가장많은 숫자로 맞춰짐
                                                         # label 간의 불균형. smote 자체적 증폭 방식은 분류에서만 사용함
print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2]), array([53, 53, 53], dtype=int64)) 증폭됨



# 2. 모델
model = RandomForestClassifier(random_state=seed, n_jobs=-1)
model.fit(x_train, y_train)

# 3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy',   
#               optimizer='adam', metrics=['acc'])

# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=30, verbose=1,
#                    restore_best_weights=True,)

# model.fit(x_train, y_train, epochs=100, validation_split=0.2,
#           callbacks=[es])

# 4. 평가, 예측
# results = model.evaluate(x_test, y_test)
# print('loss : ', results[0])
# print('acc : ', results[1])

y_pred = model.predict(x_test)
print(y_pred)
print(y_pred.shape)
y_pred = np.round(y_pred).astype(int).flatten()
# y_pred = np.argmax(y_pred, axis=1)
print(y)
print(y_pred.shape)

acc = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average='weighted') # average='macro')  # 다중은 macro
print('accuracy_score :', acc)
print('f1_score :', f1)                 

# smote 적용
# accuracy_score : 0.303860975787101
# f1_score : 0.37161984772906276

# =====================================
# ROS 적용
# accuracy_score : 0.8504326328800988
# f1_score : 0.8528765655382897