# ========================================
# 1. 라이브러리 로딩 및 환경 정보
# ========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
from xgboost import XGBClassifier, callback
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import datetime

seed = 814
random.seed(seed)
np.random.seed(seed)


# ========================================
# 2. 데이터 로딩
# ========================================
train_csv = pd.read_csv("C:/Study25/_data/dacon/갑상선암/train.csv", index_col=0)
test_csv = pd.read_csv("C:/Study25/_data/dacon/갑상선암/test.csv", index_col=0)
submission_csv = pd.read_csv("C:/Study25/_data/dacon/갑상선암/sample_submission.csv", index_col=0)



# print(train_csv)
# print(test_csv)

print(train_csv.columns)
# Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#        'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#        'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
#        'Cancer'],
#       dtype='object')
print(test_csv.columns)
# Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#        'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#        'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result'],
#       dtype='object')
print(submission_csv.columns)
# Index(['Cancer'], dtype='object')

############## 결측치 확인 ################
print(train_csv.info())
print(train_csv.isnull().sum())           # 결측치 없음
print(test_csv.isna().sum())              # 결측치 없음
print(train_csv.describe())

print(train_csv.shape, test_csv.shape)      # (87159, 15) (46204, 14)

############### train_csv와 test_csv 분리 ###############
#  1. 구분용 컬럼 추가
train_csv['is_train'] = 1
test_csv['is_train'] = 0
print(train_csv.shape, test_csv.shape)      # (87159, 16) (46204, 15)

############### 범주형 데이터 라벨인코딩 하기 #################
combined = pd.concat([train_csv, test_csv], axis=0)
print(combined)
print(combined.shape)   # (133363, 17)

aaa = pd.get_dummies(combined, columns=['Gender', 'Country', 'Race', 'Family_Background',
       'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
       'Diabetes'], drop_first=True,        # 이진 변수는 하나만 생성
                     dtype=int,)            # True/False가 아닌 1/0 정수형으로 바꿈

print(aaa)          # [133363 rows x 28 columns]
print(aaa.columns)
# Index(['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result', 'Cancer',
#        'is_train', 'is_test', 'Gender_M', 'Country_CHN', 'Country_DEU',
#        'Country_GBR', 'Country_IND', 'Country_JPN', 'Country_KOR',
#        'Country_NGA', 'Country_RUS', 'Country_USA', 'Race_ASN', 'Race_CAU',
#        'Race_HSP', 'Race_MDE', 'Family_Background_Positive',
#        'Radiation_History_Unexposed', 'Iodine_Deficiency_Sufficient',
#        'Smoke_Smoker', 'Weight_Risk_Obese', 'Diabetes_Yes'],
################# 상관계수 시작 ##################
# print(aaa.corr())

# plt.figure(figsize=(5, 12))
# sns.heatmap(aaa.corr(), annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
# plt.title("Cancer와의 상관계수 히트맵")
# plt.show()
################## 상관계수 끝 ######################

# 지울 컬럼들
# [1] ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',]      # 상관, 피처임포턴스
# [2] ['Smoke_Smoker', 'Weight_Risk_Obese', 'Diabetes_Yes']                # 상관관계는 없는데 피처임포턴스

# 그래서 우선 [1]의 컬럼 5개를 삭제한다.
drop_features = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
aaa = aaa.drop(columns=drop_features)

print(aaa)


#  4. 다시 분리
train_csv = aaa[aaa['is_train'] == 1].drop(columns='is_train')
test_csv = aaa[aaa['is_train'] == 0].drop(columns='is_train')

print(train_csv.shape, test_csv.shape)      # (87159, 22) (0, 22)
print(train_csv.columns)
print(test_csv.columns)         # Cancer 컬럼 제거
print(test_csv['Cancer'])       # 전부 NaN
test_csv = test_csv.drop(['Cancer'], axis=1)
print(train_csv.shape, test_csv.shape)           # (87159, 22) (0, 21)


####### x 와 y 분리 ##########
x = train_csv.drop(['Cancer'], axis=1)
print(x)                # [87159 rows x 21 columns]
y = train_csv['Cancer']
print(y)
print(y.shape)          # (87159,)

print(np.unique(y, return_counts=True))     # (array([0., 1.]), array([76700, 10459], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=222,
    stratify=y,
)

print(np.unique(y_train, return_counts=True))       # (array([0., 1.]), array([69030,  9413], dtype=int64))
print(np.unique(y_test, return_counts=True))        # (array([0., 1.]), array([7670, 1046], dtype=int64))

print(x_train.shape, x_test.shape)                  # (78443, 21) (8716, 21)
print(y_train.shape, y_test.shape)                  # (78443,) (8716,)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=seed)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(x_train.shape, y_train.shape)
print(pd.Series(y_train).value_counts())

print(x_train)

print(np.unique(y_train, return_counts=True))      
print(np.unique(y_test, return_counts=True))       

print(x_train.shape, x_test.shape)                 
print(y_train.shape, y_test.shape) 


# 2. 모델구성
early_stop = xgb.callback.EarlyStopping(
    rounds = 200,
    metric_name = 'logloss',            # eval_metric 과 동일하게
    data_name = 'validation_0',
    save_best = True,
    
)

model = XGBClassifier(
    n_estimators = 10000,
    max_depth = 5,
    gamma = 2,
    min_child_weight = 5,
    subsample = 0.8,
    colsample_bytree = 0.8,         # 피처 샘플링 추가
    reg_alpha = 0.1,                # L1 규제 - 가중치 규제
    reg_lambda = 1,                 # L2 규제 - 가중치 규제
    objective='binary:logistic',
    eval_metric = 'logloss',        # 다중분류:mlogloss,merror 이진분류:logloss,error /2.1.1
    # callbacks = [early_stop],
    random_state = seed,
    learning_rate=0.05,
    use_label_encoder=False,        # 재확인 요망
    callbacks = [early_stop],
    # early_stopping_rounds=50
)

# 3. 훈련
model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          # early_stopping_rounds=50,
          verbose=100,
          )

# 4. 평가, 예측
results = model.score(x_test, y_test)
(print('최종점수 :', results))

y_predict = model.predict(x_test)
print(y_predict[:10])
y_predict = np.round(y_predict)
print(y_predict[:10])

############# submission.csv 파일 만들기 // count 컬럼 값만 넣어주기
y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)

# print(submission_csv)
submission_csv['Cancer'] = y_submit
print(submission_csv[:10])

timestamp = datetime.now().strftime('%Y%m%d_%H%M')


filename = f"C:/Study25/_data/dacon/갑상선암/submission_{timestamp}.csv"
submission_csv.to_csv(filename)
print(f"Submission saved to: {filename}")

f1 = f1_score(y_test, y_predict)
print("f1 :", f1)

# 파일 저장을 위한 타임스탬프 경로 설정
weight_path = f"C:/Study25/_data/dacon/갑상선암/model_weights_{timestamp}.hdf5"
model_path = f"C:/Study25/_data/dacon/갑상선암/model_structure_{timestamp}.hdf5"


# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1615.csv
# f1 : 0.42077727952167415

# rs 222
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1633.csv
# f1 : 0.5181550539744848


#  rs 3377
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1635.csv
# f1 : 0.46703573225968803


#  rs 999
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1637.csv
# f1 : 0.4841788046207936


# # rs 1004
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1637.csv
# f1 : 0.5002433090024331

#  rs 277
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1641.csv
# f1 : 0.47159376571141276

#  rs 190
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1642.csv
# f1 : 0.5072103431128792

# rs 119
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_2007.csv
# f1 : 0.48598130841121495


# rs 190
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250626_1006.csv
# f1 : 0.4990384615384616

# rs 8787
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250627_1205.csv
# f1 : 0.5133906013137949

# rs 3377 
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250627_1450.csv
# f1 : 0.47414234511008707

# rs 3377 // 222
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250627_1451.csv
# f1 : 0.5201177625122669

#  rs 9301 //   0.5100606061	
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250630_0930.csv
# f1 : 0.518664047151277

#  rs 3569
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250630_094110.csv
# f1 : 0.5194294146581406

#  rs 1223      // 	0.5106899903	
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250630_0952.csv
# f1 : 0.5199409158050222

# rs 954    // 0.5109382596
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250630_0955.csv
# f1 : 0.5201970443349754

#  rs 7531
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250630_0957.csv
# f1 : 0.5186274509803921