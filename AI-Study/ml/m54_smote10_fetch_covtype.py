import random
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import sklearn as sk
import imblearn

# 1. Seed 고정
seed = 123
random.seed(seed)
np.random.seed(seed)

# 2. 데이터 로딩
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
y = y - 1   # 클래스 0~6으로 변환
print(np.unique(y, return_counts=True))

# 3. 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=seed, train_size=0.75, shuffle=True, stratify=y
)
print(np.unique(y_train, return_counts=True))

# 4. SMOTE 적용
smote = SMOTE(random_state=seed, sampling_strategy='auto')
x_train, y_train = smote.fit_resample(x_train, y_train)
print(np.unique(y_train, return_counts=True))

# 5. 모델 구성 및 학습
model = RandomForestClassifier(random_state=seed, n_jobs=-1)
model.fit(x_train, y_train)

# 6. 예측 및 평가
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print("accuracy_score :", acc)
print("f1_score :", f1)


# accuracy_score : 0.9577220436066725
# f1_score : 0.9367090571590853