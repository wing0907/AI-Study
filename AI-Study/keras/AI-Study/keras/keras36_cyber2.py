import pandas as pd
import numpy as np
import ipaddress
import os
import joblib
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn.covariance import EllipticEnvelope
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
path = 'C:/Study25/_data/dacon/cyber_attack/'
train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')
sample_submission = pd.read_csv(path + 'sample_submission.csv')
model_save_dir = path + 'models/'
os.makedirs(model_save_dir, exist_ok=True)

# IP 전처리 함수
def is_private_ip(ip):
    try:
        return int(ipaddress.ip_address(ip).is_private)
    except:
        return np.nan

def process_ip_columns(df, col_names):
    for col in col_names:
        df[f'{col}_private'] = df[col].apply(is_private_ip)
        mode_val = df[f'{col}_private'].mode()
        df[f'{col}_private'] = df[f'{col}_private'].fillna(mode_val[0] if not mode_val.empty else 0.0)
    return df.drop(columns=col_names)

# 데이터 전처리 함수
def preprocess_data(df, mode='train', imputer=None, scaler=None, protocol_encoder=None, attack_encoder=None):
    if 'ID' in df.columns:
        original_id = df.pop('ID')
    else:
        original_id = None
    df = process_ip_columns(df.copy(), ['ip_src', 'ip_dst'])
    if mode == 'train':
        df['protocol'] = protocol_encoder.fit_transform(df['protocol'])
    else:
        df['protocol'] = df['protocol'].apply(lambda x: x if x in protocol_encoder.classes_ else protocol_encoder.classes_[0])
        df['protocol'] = protocol_encoder.transform(df['protocol'])

    if 'attack_type' in df.columns:
        if mode == 'train':
            df['attack_type_encoded'] = attack_encoder.fit_transform(df['attack_type'])
        y = df.pop('attack_type_encoded')
        df.drop(columns=['attack_type'], inplace=True)
    else:
        y = None

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols]) if mode == 'train' else imputer.transform(df[numeric_cols])
    df_scaled = scaler.fit_transform(df) if mode == 'train' else scaler.transform(df)
    return df_scaled, y, original_id, df.columns

# 초기 설정
label_enc_attack = LabelEncoder()
label_enc_protocol = LabelEncoder()
imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, include_bias=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 데이터 전처리 및 다항 특성 생성
X_train_raw, y_train, _, feature_names = preprocess_data(train_df.copy(), 'train', imputer, scaler, label_enc_protocol, label_enc_attack)
X_test_raw, _, test_ids, _ = preprocess_data(test_df.copy(), 'test', imputer, scaler, label_enc_protocol)
X_train_poly = poly.fit_transform(X_train_raw)
X_test_poly = poly.transform(X_test_raw)
n_classes = len(np.unique(y_train))

# 저장용 변수
oof_preds = np.zeros((X_train_poly.shape[0], n_classes))
oof_labels = np.zeros(X_train_poly.shape[0])
test_preds_list = []

# K-Fold 학습 및 저장
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_poly, y_train)):
    print(f"\n===== Fold {fold+1} =====")
    X_tr, X_val = X_train_poly[train_idx], X_train_poly[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # 이상치 제거
    outlier_detector = EllipticEnvelope(contamination=0.01, random_state=707)
    inlier_mask = outlier_detector.fit_predict(X_tr) == 1
    X_tr, y_tr = X_tr[inlier_mask], y_tr[inlier_mask]

    # SMOTE 적용
    original_dist = Counter(y_tr)
    minority = [cls for cls, cnt in original_dist.items() if cnt <= 100]
    target_dist = {cls: min(cnt * 3, 300) if cls in minority else cnt for cls, cnt in original_dist.items()}
    smote = SMOTE(sampling_strategy=target_dist, random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X_tr, y_tr)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_res), y=y_res)
    sample_weights = pd.Series(y_res).map(dict(enumerate(class_weights))).values

    # 모델 정의 및 학습
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, objective='multi:softprob',
                        num_class=n_classes, eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    lgb = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, objective='multiclass',
                         num_class=n_classes, metric='multi_logloss', random_state=42, verbosity=-1, force_col_wise=True)
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, loss_function='MultiClass',
                             random_seed=42, verbose=False)

    xgb.fit(X_res, y_res, sample_weight=sample_weights)
    lgb.fit(X_res, y_res, sample_weight=sample_weights)
    cat.fit(X_res, y_res, sample_weight=sample_weights)

    joblib.dump(xgb, f"{model_save_dir}xgb_fold{fold+1}.pkl")
    joblib.dump(lgb, f"{model_save_dir}lgb_fold{fold+1}.pkl")
    cat.save_model(f"{model_save_dir}cat_fold{fold+1}.cbm")

    val_proba = (xgb.predict_proba(X_val) + lgb.predict_proba(X_val) + cat.predict_proba(X_val)) / 3
    oof_preds[val_idx] = val_proba
    oof_labels[val_idx] = y_val

    test_proba = (xgb.predict_proba(X_test_poly) + lgb.predict_proba(X_test_poly) + cat.predict_proba(X_test_poly)) / 3
    test_preds_list.append(test_proba)

# OOF 평가
final_oof_preds = np.argmax(oof_preds, axis=1)
acc = accuracy_score(oof_labels, final_oof_preds)
f1_macro = f1_score(oof_labels, final_oof_preds, average='macro')
f1_weighted = f1_score(oof_labels, final_oof_preds, average='weighted')
print(f"\n===== OOF Performance =====")
print(f"Accuracy:     {acc:.4f}")
print(f"Macro F1:     {f1_macro:.4f}")
print(f"Weighted F1:  {f1_weighted:.4f}")

# 제출 파일 생성
final_test_proba = np.mean(test_preds_list, axis=0)
final_preds = np.argmax(final_test_proba, axis=1)
final_labels = label_enc_attack.inverse_transform(final_preds)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
submission = pd.DataFrame({'ID': test_ids, 'attack_type': final_labels})
submission_path = path + f'submission_{timestamp}.csv'
submission.to_csv(submission_path, index=False)
print(f"\n📁 제출 파일 저장 완료: {submission_path}")
print(submission.head())


# ===== OOF Performance =====
# Accuracy:     0.9924
# Macro F1:     0.8338
# Weighted F1:  0.9925

# 📁 제출 파일 저장 완료: C:/Study25/_data/dacon/cyber_attack/submission_20250718_181612.csv
#           ID    attack_type
# 0  TEST_0000         Benign
# 1  TEST_0001  Port_Scanning
# 2  TEST_0002         Benign
# 3  TEST_0003         Benign
# 4  TEST_0004         Benign

# ===== OOF Performance =====
# Accuracy:     0.9925
# Macro F1:     0.8342
# Weighted F1:  0.9926

# 📁 제출 파일 저장 완료: C:/Study25/_data/dacon/cyber_attack/submission_20250718_193040.csv
#           ID    attack_type
# 0  TEST_0000         Benign
# 1  TEST_0001  Port_Scanning
# 2  TEST_0002         Benign
# 3  TEST_0003         Benign
# 4  TEST_0004         Benign