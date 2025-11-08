import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from xgboost import XGBClassifier

# Seed
seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터 로드
path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# target → 숫자형 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 'Class_1' → 0, ..., 'Class_9' → 8

# 스케일링
scaler = RobustScaler()
x_scaled = scaler.fit_transform(x)

# train/test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded
)

# 모델 학습
model = XGBClassifier(random_state=seed)
model.fit(x_train, y_train)
print("🔥 제거 전 acc:", model.score(x_test, y_test))

# Feature importance 기준 하위 25% 제거
importances = model.feature_importances_
threshold = np.percentile(importances, 25)
low_idx = np.where(importances <= threshold)[0]

# 불필요한 feature 제거
x_reduced = pd.DataFrame(x_scaled).drop(columns=low_idx)

# 다시 데이터 분리
x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(
    x_reduced, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded
)

# 재학습
model.fit(x_train_r, y_train_r)
print("✅ 제거 후 acc:", model.score(x_test_r, y_test_r))

# 🔥 제거 전 acc: 0.8116515837104072
# ✅ 제거 후 acc: 0.8106819650937298