import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# 시드 고정
seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터 불러오기
path = 'C:/Study25/_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Feature, Target 분리
x = train_csv.drop('target', axis=1)
y = train_csv['target']
print(np.unique(y, return_counts=True))  # [0 1], [179902 20098] → 불균형

# 3. Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, random_state=seed, stratify=y
)
print('Before SMOTE:', np.unique(y_train, return_counts=True))

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

x_train, y_train = ros.fit_resample(x_train, y_train) 


# 5. 머신러닝 모델 학습
model = RandomForestClassifier(random_state=seed, n_jobs=-1)
model.fit(x_train, y_train)

# 6. 예측 및 평가
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print('accuracy_score :', acc)
print('f1_score :', f1)

# accuracy_score : 0.8941
# f1_score : 0.06432231843081816

# =====================================
# ROS 적용
# Before SMOTE: (array([0, 1], dtype=int64), array([134926,  15074], dtype=int64))
# sklearn version :  1.6.1
# imblearn version : 0.12.4
# accuracy_score : 0.89952
# f1_score : 0.8519375741239892