import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)                     # (178, 13) (178,)
print(np.unique(y, return_counts=True))     # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
print(y)

### 데이터 삭제. 라벨이 2인놈을 10개만 남기고 다 지워라!!! 리스트의 슬라이싱으로 지우기
# [59, 71, 8]
x = x[:-40]
y = y[:-40]
print(y)

print(np.unique(y, return_counts=True))     # (array([0, 1, 2]), array([59, 71,  8], dtype=int64))

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
            #   sampling_strategy='auto', # default. 머리쓰기 싫을땐 디폴트가 아름답다.
            #   sampling_strategy=0.75, # 최대값의 75% 지점
              sampling_strategy={0:500, 2:500, 1:500}, # 이렇게 하면 1.0 나옴. 아름다운 데이터일 경우.
            #   n_jobs=-1, # 0.13에서는 삭제됨. 기냥 포함됨          
              
              )

# smote = SMOTE(random_state=seed,
#               k_neighbors=5, # default
#             #   sampling_strategy='auto', # default. 머리쓰기 싫을땐 디폴트가 아름답다.
#             #   sampling_strategy=0.75, # 최대값의 75% 지점
#               sampling_strategy={0:500, 2:500, 1:500}, # 이렇게 하면 1.0 나옴. 아름다운 데이터일 경우.
#               n_jobs=-1, # 0.13에서는 삭제됨. 기냥 포함됨          
#               )

x_train, y_train = ros.fit_resample(x_train, y_train)  # smote.fit_resample 가장많은 숫자로 맞춰짐
                                                         # label 간의 불균형. smote 자체적 증폭 방식은 분류에서만 사용함
print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2]), array([53, 53, 53], dtype=int64)) 증폭됨



# 2. 모델
model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',   #원핫 안했으면 요놈
              optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

y_pred = model.predict(x_test)
print(y_pred)
print(y_pred.shape)
y_pred = np.argmax(y_pred, axis=1)
print(y)
print(y_pred.shape)

acc = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average='macro')  # 다중은 macro
print('accuracy_score :', acc)
print('f1_score :', f1)                 


# 슬라이싱 한거
# accuracy_score : 0.8857142857142857
# f1_score : 0.615890083632019

# 원데이터
# accuracy_score : 0.9333333333333333
# f1_score : 0.9252905078992035

# smote 적용
# accuracy_score : 0.9142857142857143
# f1_score : 0.6354354354354355

# ========================================

# ROS 적용
# accuracy_score : 0.9428571428571428
# f1_score : 0.9029728020240354