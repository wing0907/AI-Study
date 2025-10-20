# https://www.kaggle.com/competitions/playground-series-s4e1/submissions

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
import time

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
'''
le = LabelEncoder()                 # 함수를 인스턴스화 한다
train_csv['Geography'] = le.fit_transform(train_csv['Geography'])       # le 를 train_csv에 있는 Geography 컬럼을 적용해서 변환시키겠다 = b에 집어넣겠다
train_csv['Gender'] = le.fit_transform(train_csv['Gender']) 
####################################################################

le = LabelEncoder()
le.fit(train_csv['Geography'])
train_csv['Geography'] = le.transform(train_csv['Geography'])
print(train_csv['Geography'].value_counts())

le.fit(train_csv['Gender'])
train_csv['Gender'] = le.transform(train_csv['Gender'])
'''
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
print(train_csv.columns)
# ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
#        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
#        'Exited']


x = train_csv.drop(['Exited'], axis=1)
print(x.shape)      # (165034, 10)
y = train_csv['Exited']
print(y.shape)      # (165034,)

# scaler = StandardScaler()
# x_scaled = scaler.fit_transform(x)
# test_scaled = scaler.transform(test_csv)

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


# from sklearn.preprocessing import MinMaxScaler

# # 1. 컬럼 분리
# x_other = x.drop(['EstimatedSalary'], axis=1)
# x_salary = x[['EstimatedSalary']]

# # 2. 각각 스케일링
# scaler_other = MinMaxScaler()                    # 기본 [0, 1]
# scaler_salary = MinMaxScaler(feature_range=(0, 100))  # 지정 범위

# x_other_scaled = scaler_other.fit_transform(x_other)
# x_salary_scaled = scaler_salary.fit_transform(x_salary)

# # 3. 합치기
# x_scaled = np.concatenate([x_other_scaled, x_salary_scaled], axis=1)

# # 4. test set도 동일하게 처리
# test_other = test_csv.drop(['EstimatedSalary'], axis=1)
# test_salary = test_csv[['EstimatedSalary']]

# test_other_scaled = scaler_other.transform(test_other)
# test_salary_scaled = scaler_salary.transform(test_salary)

# test_scaled = np.concatenate([test_other_scaled, test_salary_scaled], axis=1)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_predict = scaler.transform(x_predict)

'''
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x['EstimatedSalary'] = scaler.fit_transform(x[['EstimatedSalary']])        # train data에 맞춰서 스케일링
test_csv['EstimatedSalary'] = scaler.transform(test_csv[['EstimatedSalary']])  # test data는 transform만

x_scaled = x.values          # 넘파이 배열로 변환
test_scaled = test_csv.values
'''

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size=0.8, random_state=588
)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))


# 2. 모델구성
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(128, input_dim=x_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])



# 3. 컴파일, 훈련
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()
print(date)     # 2025-06-02 13:00:44.718308
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")     # string 문자열
print(date)     # 0602_1305
print(type(date))   # <class 'str'>

path = './_save/keras28_mcp/08_kaggle_bank/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k28_', date, '_', filename])



mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath
)



start_time = time.time()

hist = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    # validation_split=0.2,
    epochs=1000,
    batch_size=512,
    callbacks=[es, mcp],
    class_weight=class_weights,
    verbose=1
)
end_time = time.time()

print('=========  hist  ========')
print(hist)     # <keras.callbacks.History object at 0x0000022D52E9AC10>
print('=========  history  ========')
print(hist.history)
print('=========  loss  ========') #loss 값만 보고 싶을 경우.
print(hist.history['loss'])
print('=========  val_loss  ========')
print(hist.history['val_loss'])
print('=========  val_accuracy  ========')
print(hist.history['val_accuracy'])

# 그래프 그리기
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
matplotlib.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=(9,6))       # 9 x 6 사이즈
plt.plot(hist.history['loss'], c='red', label='loss')  # y값만 넣으면 시간순으로 그린 그림
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.plot(hist.history['val_accuracy'], c='green', label='val_accuracy')
plt.title('KAGGLE BANK Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')       #우측 상단에 label 표시
plt.grid()                          #격자 표시
plt.show()



# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('Evaluation - Loss:', (loss, 4), 'Accuracy:', round(acc, 4))

# Predict on test split
y_pred_prob = model.predict(x_test)
threshold = 0.45
y_pred_binary = (y_pred_prob > threshold).astype(int)
acc_score = accuracy_score(y_test, y_pred_binary)
print(f"Threshold={threshold}, Accuracy={round(acc_score, 4)}")


print("acc_score : ", acc_score)
print("걸린시간 : ", round(end_time - start_time, 2), "초")   


# Predict on submission set
y_submit = model.predict(test_scaled)
# y_submit = np.round(y_submit)


submission_csv['Exited'] = y_submit
# submission_csv.to_csv(path + 'submission_0528_1430.csv')
# print("Submission saved as 'submission_0528_1430.csv'")

# Evaluation - Loss: (0.41097885370254517, 4) Accuracy: 0.8151
# Threshold=0.45, Accuracy=0.7954
# acc_score :  0.795407034871391
# 걸린시간 :  65.17 초