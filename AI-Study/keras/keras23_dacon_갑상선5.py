import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
from imblearn.over_sampling import SMOTE


# =============================
# 1. 데이터 로드
# =============================
train_df = pd.read_csv("C:/Study25/_data/dacon/갑상선암/train.csv", index_col=0)
test_df = pd.read_csv("C:/Study25/_data/dacon/갑상선암/test.csv", index_col=0)
submission_df = pd.read_csv("C:/Study25/_data/dacon/갑상선암/sample_submission.csv", index_col=0)

# =============================
# 2. 데이터 분할 및 전처리
# =============================

# 결측치 제거
train_df = train_df.dropna()
test_df = test_df.dropna()

x = train_df.drop("Cancer", axis=1)
y = train_df["Cancer"]
test_x = test_df.copy()

# 이상치 처리 (Z-score 기준으로 3.0 초과 값 제거)
from scipy.stats import zscore
z_scores = np.abs(zscore(x.select_dtypes(include=[np.number])))
filtered_entries = (z_scores < 3).all(axis=1)
x = x[filtered_entries]
y = y[filtered_entries]

# 범주형 변수 처리 (Label Encoding)
categorical_cols = x.select_dtypes(include='object').columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    encoders[col] = le

for col in categorical_cols:
    le = encoders[col]
    test_x[col] = test_x[col].map(lambda s: '<UNK>' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '<UNK>')
    test_x[col] = le.transform(test_x[col])

# 수치형 스케일링
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_x)

# Feature importance 제거 및 재분할은 이미 반영됨

# 중요도 낮은 피처 제거
important_drop_features = ["T3_Result", "Nodule_Size", "Age", "T4_Result", "TSH_Result"]
x = pd.DataFrame(x_scaled, columns=x.columns).drop(columns=important_drop_features)
test_scaled = pd.DataFrame(test_scaled, columns=test_x.columns).drop(columns=important_drop_features)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=80, stratify=y)

smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

# =============================
# 5. Optuna 튜닝 및 앙상블 모델 학습 (VotingClassifier)
# =============================

def objective_xgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 190
    }
    model = XGBClassifier(**params)
    model.fit(x_train_res, y_train_res)
    preds = model.predict(x_val)
    return f1_score(y_val, preds)

def objective_lgbm(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'random_state': 190
    }
    model = LGBMClassifier(**params)
    model.fit(x_train_res, y_train_res)
    preds = model.predict(x_val)
    return f1_score(y_val, preds)

def objective_cat(trial):
    params = {
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'iterations': trial.suggest_int('iterations', 100, 500),
        'random_state': 190,
        'verbose': 0
    }
    model = CatBoostClassifier(**params)
    model.fit(x_train_res, y_train_res)
    preds = model.predict(x_val)
    return f1_score(y_val, preds)

print("[Optuna] Tuning XGBoost...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50)
print("Best XGB Params:", study_xgb.best_params)

print("[Optuna] Tuning LightGBM...")
study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=50)
print("Best LGBM Params:", study_lgbm.best_params)

print("[Optuna] Tuning CatBoost...")
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=50)
print("Best CatBoost Params:", study_cat.best_params)

# 최적 파라미터 기반 모델 재정의
xgb = XGBClassifier(**study_xgb.best_params, eval_metric='logloss', use_label_encoder=False, random_state=42)
lgbm = LGBMClassifier(**study_lgbm.best_params, random_state=190)
cat = CatBoostClassifier(**study_cat.best_params, random_state=190, verbose=0)

ensemble_model = VotingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
    voting='soft'
)
ensemble_model.fit(x_train_res, y_train_res)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")


# =============================
# 6. Threshold 튜닝 기반 예측 및 제출
# =============================
from sklearn.metrics import precision_recall_curve

val_proba = ensemble_model.predict_proba(x_val)[:, 1]
prec, rec, thresh = precision_recall_curve(y_val, val_proba)
f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
best_idx = np.argmax(f1)
best_thresh = thresh[best_idx]
val_pred_thresh = (val_proba > best_thresh).astype(int)

print(f"Best Threshold: {best_thresh:.2f} | Validation F1 Score (thresholded): {f1[best_idx]:.4f}")

test_proba = ensemble_model.predict_proba(test_scaled)[:, 1]
test_pred = (test_proba > best_thresh).astype(int)
submission_df['Cancer'] = test_pred

filename = f"C:/Study25/_data/dacon/갑상선암/submission_{timestamp}.csv"
submission_df.to_csv(filename)
print(f"Submission saved: {filename}")
