# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/data
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.utils import class_weight



import time

# 1. 데이터
path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)


print(train_csv.isna().sum())      
print(test_csv.isna().sum()) 

print(train_csv.columns)
# Index(['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6', 'feat_7',
#        'feat_8', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'feat_13',
#        'feat_14', 'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19',
#        'feat_20', 'feat_21', 'feat_22', 'feat_23', 'feat_24', 'feat_25',
#        'feat_26', 'feat_27', 'feat_28', 'feat_29', 'feat_30', 'feat_31',
#        'feat_32', 'feat_33', 'feat_34', 'feat_35', 'feat_36', 'feat_37',
#        'feat_38', 'feat_39', 'feat_40', 'feat_41', 'feat_42', 'feat_43',
#        'feat_44', 'feat_45', 'feat_46', 'feat_47', 'feat_48', 'feat_49',
#        'feat_50', 'feat_51', 'feat_52', 'feat_53', 'feat_54', 'feat_55',
#        'feat_56', 'feat_57', 'feat_58', 'feat_59', 'feat_60', 'feat_61',
#        'feat_62', 'feat_63', 'feat_64', 'feat_65', 'feat_66', 'feat_67',
#        'feat_68', 'feat_69', 'feat_70', 'feat_71', 'feat_72', 'feat_73',
#        'feat_74', 'feat_75', 'feat_76', 'feat_77', 'feat_78', 'feat_79',
#        'feat_80', 'feat_81', 'feat_82', 'feat_83', 'feat_84', 'feat_85',
#        'feat_86', 'feat_87', 'feat_88', 'feat_89', 'feat_90', 'feat_91',
#        'feat_92', 'feat_93', 'target'],
#       dtype='object')
print(train_csv)        # [61878 rows x 94 columns]
print(test_csv)         # [144368 rows x 93 columns]
print(train_csv.shape)  # (61878, 94)
print(test_csv.shape)   # (144368, 93)
print(submission_csv.shape) # (144368, 9)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

print(x.shape, y.shape)     # (61878, 93) (61878,)
print(np.unique(y, return_counts=True))
# (array(['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6',
#        'Class_7', 'Class_8', 'Class_9'], dtype=object), array([ 1929, 16122,  8004,  2691,  2739, 14135,  2839,  8464,  4955],
#       dtype=int64))
print(pd.DataFrame(y).value_counts())
# target
# Class_2    16122
# Class_6    14135
# Class_8     8464
# Class_3     8004
# Class_9     4955
# Class_7     2839
# Class_5     2739
# Class_4     2691
# Class_1     1929
# dtype: int64



# 1. 문자열 배열 → numpy로 변환
# y = train_csv['target'].values.reshape(-1, 1)  # (61878, 1)
# print(y.shape)
# 2. One-Hot 인코딩
# ohe = OneHotEncoder(sparse=False)  # → sparse matrix 말고 array로 받기
# y_encoded = ohe.fit_transform(y)   # (61878, 9)

# print(y_encoded.shape)  # (61878, 9)
# print(type(y_encoded))  # <class 'numpy.ndarray'>

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



#  2. 모델구성
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(128, input_dim=93, activation='relu'),
    BatchNormalization(),
    # Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(9, activation='softmax')
])


#  3. compile, fit
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


es = EarlyStopping(
    monitor= 'val_loss',
    mode='min',
    patience= 30,
    restore_best_weights=True
)


import datetime
date = datetime.datetime.now()
print(date)     # 2025-06-02 13:00:44.718308
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")     # string 문자열
print(date)     # 0602_1305
print(type(date))   # <class 'str'>

import os

os.makedirs('./_save/keras30_scaler/otto/', exist_ok=True)
filepath = f"./_save/keras30_scaler/otto/k30_otto_{date}_{{epoch:04d}}-{{val_loss:.4f}}.hdf5"


mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath
)

start_time = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=1000,
          batch_size=512,
          validation_split=0.2,
          verbose=1,
          class_weight=class_weights,
          callbacks=[es, mcp])

end_time = time.time()




#  4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('Evaluation - Loss:', (loss, 4), 'Accuracy:', round(acc, 4))


print("걸린시간 : ", round(end_time - start_time, 2), "초")  

test_scaled = scaler.transform(test_csv)

# 예측 결과: (144368, 9) → softmax 확률값
y_submit = model.predict(test_scaled)

# sampleSubmission.csv의 id 유지
submission = pd.DataFrame(y_submit,
                          columns=['Class_1', 'Class_2', 'Class_3', 'Class_4',
                                   'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'],
                          index=submission_csv.index)

# 'id' 열 붙이기
submission.insert(0, 'id', submission_csv.index)

# 저장
submission.to_csv(f'./_data/kaggle/otto/submission_{date}.csv', index=False)
print(f"✅ 제출 파일 저장 완료: submission_{date}.csv")

import tensorflow as tf
print(tf.__version__) # 2.9.3           tensorflow는 2.10 버전까지는 gpu와 cpu 버전이 따로 나누어져있다

gpus =tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')
    
# Evaluation - Loss: (0.5869044065475464, 4) Accuracy: 0.7673
# 걸린시간 :  69.8 초
# ✅ 제출 파일 저장 완료: submission_0609_0144.csv

# cpu
# Evaluation - Loss: (0.5717079639434814, 4) Accuracy: 0.7751
# 걸린시간 :  90.88 초
# ✅ 제출 파일 저장 완료: submission_0609_0149.csv

# Evaluation - Loss: (0.5806244611740112, 4) Accuracy: 0.7745
# 걸린시간 :  19.38 초
# ✅ 제출 파일 저장 완료: submission_0609_0937.csv

##############################################################
# gpu
# Evaluation - Loss: (0.5713947415351868, 4) Accuracy: 0.7771
# 걸린시간 :  45.67 초
# ✅ 제출 파일 저장 완료: submission_0609_0922.csv

# Evaluation - Loss: (0.5757341980934143, 4) Accuracy: 0.7761
# 걸린시간 :  42.41 초
# ✅ 제출 파일 저장 완료: submission_0609_0952.csv