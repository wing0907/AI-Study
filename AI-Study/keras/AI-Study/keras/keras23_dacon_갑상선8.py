# ========================================
# 1. 라이브러리 로딩 및 환경 정보
# ========================================
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

# 파일 저장을 위한 타임스탬프 경로 설정
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
weight_path = f"C:/Study25/_data/dacon/갑상선암/model_weights_{timestamp}.hdf5"
model_path = f"C:/Study25/_data/dacon/갑상선암/model_structure_{timestamp}.hdf5"

# ========================================
# 2. 데이터 로딩
# ========================================
train = pd.read_csv("C:/Study25/_data/dacon/갑상선암/train.csv", index_col=0)
test = pd.read_csv("C:/Study25/_data/dacon/갑상선암/test.csv", index_col=0)
submission = pd.read_csv("C:/Study25/_data/dacon/갑상선암/sample_submission.csv", index_col=0)

# ========================================
# 3. 전처리: 범주형 인코딩만 적용
# ========================================

# 불필요한 수치형 파생 변수 제거 및 범주형 get_dummies 준비
categorical = train.select_dtypes(include='object').columns

all_data = pd.concat([train.drop('Cancer', axis=1), test], axis=0)

# pd.get_dummies() 적용
all_data = pd.get_dummies(all_data, columns=categorical, drop_first=True)
# ========================================

categorical = train.select_dtypes(include='object').columns
all_data = pd.concat([train.drop('Cancer', axis=1), test], axis=0)

# pd.get_dummies() 적용
all_data = pd.get_dummies(all_data, columns=categorical, drop_first=True)

# 다시 분리
x = all_data.iloc[:len(train), :]
test_x = all_data.iloc[len(train):, :]
y = train['Cancer']
x = all_data.iloc[:len(train), :]
test_x = all_data.iloc[len(train):, :]
y = train['Cancer']


# ========================================
# 4. 중요도 시각화 전용 모델 학습
# ========================================
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=55, stratify=y)
xgb_temp = XGBClassifier(random_state=55)
xgb_temp.fit(x_train, y_train)

# ========================================
# 5. Feature Importance 시각화
# ========================================
importances = xgb_temp.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=x.columns[indices])
plt.title("Feature Importances from XGB")
plt.tight_layout()
plt.show()

# ========================================
# 6. 중요도 낮은 feature 제거
# ========================================
important_features = x.columns[importances > 0.03]
x = x[important_features]
test_x = test_x[important_features]


# ========================================
# 7. 모델 학습 (XGBoost 단일 모델)
# ========================================
x_scaled = x
test_scaled = test_x
x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=55)

xgb = XGBClassifier(
    scale_pos_weight=7.3,
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=55,
    use_label_encoder=False,
    eval_metric='logloss'
)



xgb_only = XGBClassifier(
    scale_pos_weight=7.3,
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=55,
    use_label_encoder=False,
    eval_metric='logloss'
)


xgb_only.fit(x_train, y_train)
val_pred_proba = xgb_only.predict_proba(x_val)[:, 1]

# ========================================
# 8. Threshold 최적화
# ========================================
best_f1, best_thresh = 0, 0.5
for t in np.arange(0.05, 0.91, 0.01):
    score = f1_score(y_val, (val_pred_proba > t).astype(int))
    if score > best_f1:
        best_f1, best_thresh = score, t
print(f"Best Threshold: {best_thresh:.2f} | F1: {best_f1:.4f}")

# ========================================
# 9. 테스트셋 예측 및 제출 저장
# ========================================
test_pred = (xgb_only.predict_proba(test_scaled)[:, 1] > best_thresh).astype(int)
submission['Cancer'] = test_pred
filename = f"C:/Study25/_data/dacon/갑상선암/submission_{timestamp}.csv"
submission.to_csv(filename)
print(f"Submission saved to: {filename}")



# Best Threshold: 0.61 | F1: 0.5088
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1212.csv


# Best Threshold: 0.50 | F1: 0.5109
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1215.csv


# Best Threshold: 0.44 | F1: 0.5183
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1228.csv


# Best Threshold: 0.44 | F1: 0.5183
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250626_1236.csv

# Best Threshold: 0.44 | F1: 0.5183
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250627_1813.csv

# Best Threshold: 0.45 | F1: 0.4852
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250627_1815.csv