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
# 3. 전처리: 파생변수 생성

# 불필요한 파생 변수 제거 (안정성 확보)
for col in ['Radiation_Nodule']:
    if col in train.columns:
        train.drop(columns=col, inplace=True)
    if col in test.columns:
        test.drop(columns=col, inplace=True)

# 예시: TSH, T3, T4 합산한 새로운 변수 생성
if {'TSH_Result', 'T3_Result', 'T4_Result'}.issubset(train.columns):
    train['TSH'] = train['TSH_Result'] + train['T3_Result'] + train['T4_Result']
    test['TSH'] = test['TSH_Result'] + test['T3_Result'] + test['T4_Result']


# 예시 2: 나이 기반 고령자 구분 변수 (컬럼 존재 시)
if 'Age' in train.columns:
    train['Is_Senior'] = (train['Age'] > 60).astype(int)
    test['Is_Senior'] = (test['Age'] > 60).astype(int)



# 추가 파생 변수: 상대 비율 및 구간화 및 조합

# 조합 파생변수
if {'T3_Result', 'T4_Result'}.issubset(train.columns):
    train['T3_T4_Diff'] = train['T3_Result'] - train['T4_Result']
    test['T3_T4_Diff'] = test['T3_Result'] - test['T4_Result']

if {'TSH_Result', 'Age'}.issubset(train.columns):
    train['TSH_Age_Interaction'] = train['TSH_Result'] * train['Age']
    test['TSH_Age_Interaction'] = test['TSH_Result'] * test['Age']

if {'TSH_Ratio', 'Is_Senior'}.issubset(train.columns):
    train['TSHRatio_Senior_Interaction'] = train['TSH_Ratio'] * train['Is_Senior']
    test['TSHRatio_Senior_Interaction'] = test['TSH_Ratio'] * test['Is_Senior']
if {'TSH_Result', 'T4_Result'}.issubset(train.columns):
    train['TSH_Ratio'] = train['TSH_Result'] / (train['T4_Result'] + 1e-6)
    test['TSH_Ratio'] = test['TSH_Result'] / (test['T4_Result'] + 1e-6)

if 'Age' in train.columns:
    train['Age_Bin'] = pd.cut(train['Age'], bins=[0, 30, 50, 70, 100], labels=['Young', 'Middle', 'Senior', 'Elder']).astype(str)
    test['Age_Bin'] = pd.cut(test['Age'], bins=[0, 30, 50, 70, 100], labels=['Young', 'Middle', 'Senior', 'Elder']).astype(str)

# ========================================
# 3. 전처리: pd.get_dummies()로 범주형 인코딩
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
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=7431, stratify=y)
xgb_temp = XGBClassifier(random_state=222)
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
important_features = x.columns[importances > 0.02]
x = x[important_features]
test_x = test_x[important_features]


# ========================================
# 7. 모델 학습 (앙상블: XGB + LGBM + CatBoost)
# ========================================
x_scaled = x
test_scaled = test_x
x_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=7431)

xgb = XGBClassifier(
    scale_pos_weight=7.3,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=222,
    use_label_encoder=False,
    eval_metric='logloss'
)
lgbm = LGBMClassifier(
    class_weight='balanced',
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.8,
    random_state=222
)
cat = CatBoostClassifier(
    class_weights=[1, 7.3],
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.7,
    verbose=0,
    random_state=222
)

voting = VotingClassifier(estimators=[
    ('xgb', xgb),
    ('lgbm', lgbm),
    ('cat', cat)
], voting='soft', weights=[1, 2, 3])

voting.fit(x_train, y_train)
val_pred_proba = voting.predict_proba(x_val)[:, 1]

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
test_pred = (voting.predict_proba(test_scaled)[:, 1] > best_thresh).astype(int)
submission['Cancer'] = test_pred
filename = f"C:/Study25/_data/dacon/갑상선암/submission_{timestamp}.csv"
submission.to_csv(filename)
print(f"Submission saved to: {filename}")


# Best Threshold: 0.24 | F1: 0.4682

# Best Threshold: 0.24 | F1: 0.5102

# Best Threshold: 0.12 | F1: 0.5109
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1022.csv

# Best Threshold: 0.13 | F1: 0.5109
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1038.csv

# Best Threshold: 0.13 | F1: 0.5109
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1052.csv

# Best Threshold: 0.12 | F1: 0.5109
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1057.csv

# Best Threshold: 0.13 | F1: 0.5109
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1106.csv

# Best Threshold: 0.20 | F1: 0.5109
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1138.csv


# Best Threshold: 0.23 | F1: 0.5112    500
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1142.csv

# Best Threshold: 0.22 | F1: 0.5102     1000
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1143.csv

# Best Threshold: 0.22 | F1: 0.5107     700
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1145.csv

# Best Threshold: 0.22 | F1: 0.5110     550
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1146.csv

# Best Threshold: 0.23 | F1: 0.5110     450
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1147.csv

# Best Threshold: 0.67 | F1: 0.5112
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1202.csv

# Best Threshold: 0.56 | F1: 0.5179     rs 222      ######################    54등    ###################
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1240.csv

# Best Threshold: 0.52 | F1: 0.5122
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1251.csv

# Best Threshold: 0.59 | F1: 0.5180
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1254.csv

# Best Threshold: 0.59 | F1: 0.5183
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1302.csv

# Best Threshold: 0.70 | F1: 0.5185
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1308.csv

# Best Threshold: 0.67 | F1: 0.5187             ##########################   50등 ###############################
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1314.csv


# Best Threshold: 0.44 | F1: 0.5183
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1315.csv

# Best Threshold: 0.44 | F1: 0.5183
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250625_1318.csv


# Best Threshold: 0.43 | F1: 0.5183     rs 222
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250626_1002.csv


# Best Threshold: 0.44 | F1: 0.4802     rs 3377
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250627_1817.csv

# Best Threshold: 0.44 | F1: 0.5183     rs 222
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250627_1820.csv

# Best Threshold: 0.52 | F1: 0.4773     rs 7431
# Submission saved to: C:/Study25/_data/dacon/갑상선암/submission_20250627_1824.csv