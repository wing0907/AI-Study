import pandas as pd
import numpy as np
import ipaddress
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score
from scipy.special import softmax # For soft voting

path = 'C:\Study25\_data\dacon\cyber_attack\\'
train_dir = path+'train.csv'
test_dir = path+'test.csv'
sample_submission_dir = path+'sample_submission.csv'

# 데이터 로드
train_df = pd.read_csv(train_dir)
test_df = pd.read_csv(test_dir)
sample_submission = pd.read_csv(sample_submission_dir)

### 1. 데이터 전처리 함수 (재사용성 및 효율성 증대) ###

# 사설 IP 여부 판단 함수
def is_private_ip(ip):
    try:
        return int(ipaddress.ip_address(ip).is_private)
    except:
        return np.nan # IP가 잘못됐거나 null일 때

# IP 컬럼 처리 함수
def process_ip_columns(df, col_names):
    for col in col_names:
        df[f'{col}_private'] = df[col].apply(is_private_ip)
        # 최빈값 구해서 결측치 대체
        mode_val = df[f'{col}_private'].mode()
        if not mode_val.empty:
            df[f'{col}_private'] = df[f'{col}_private'].fillna(mode_val[0])
        else:
            df[f'{col}_private'] = df[f'{col}_private'].fillna(0.0) # fallback for empty mode
    df = df.drop(columns=col_names)
    return df

# 전처리 파이프라인 함수
def preprocess_data(df, mode='train', imputer=None, scaler=None, protocol_encoder=None, attack_encoder=None):
    original_id = None
    if 'ID' in df.columns:
        original_id = df['ID']
        df = df.drop(columns=['ID'])

    # IP 주소 처리
    df = process_ip_columns(df.copy(), ['ip_src', 'ip_dst'])

    # 프로토콜 라벨 인코딩 (CatBoost는 내부 처리 가능하나, 다른 모델을 위해 일단 인코딩)
    if mode == 'train':
        df['protocol'] = protocol_encoder.fit_transform(df['protocol'])
    else: # test mode
        # unseen label 방지
        df['protocol'] = df['protocol'].apply(lambda x: x if x in protocol_encoder.classes_ else protocol_encoder.classes_[0])
        df['protocol'] = protocol_encoder.transform(df['protocol'])

    # 공격 유형 라벨 인코딩
    if 'attack_type' in df.columns:
        if mode == 'train':
            df['attack_type_encoded'] = attack_encoder.fit_transform(df['attack_type'])
        else: # test mode (test 셋에는 attack_type이 없어야 함)
            pass # test 셋에는 attack_type이 없으므로 인코딩할 필요 없음
        y = df['attack_type_encoded']
        X = df.drop(columns=['attack_type', 'attack_type_encoded'])
    else:
        y = None
        X = df

    # 숫자형 컬럼만 선택하여 결측치 대체
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if mode == 'train':
        imputed_data = imputer.fit_transform(X[numeric_cols])
    else:
        imputed_data = imputer.transform(X[numeric_cols])
    X[numeric_cols] = pd.DataFrame(imputed_data, columns=numeric_cols, index=X.index)

    # 스케일링
    if mode == 'train':
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    if original_id is not None:
        return X_scaled, y, original_id, X.columns # X.columns를 반환하여 CatBoost에서 categorical_features 지정에 활용
    else:
        return X_scaled, y, X.columns # X.columns를 반환하여 CatBoost에서 categorical_features 지정에 활용

### 2. 데이터 전처리 및 초기화 ###

# 인코더 및 스케일러 초기화 (공통 사용)
attack_label_encoder = LabelEncoder()
protocol_label_encoder = LabelEncoder()
imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()

X_train_processed, y_train_encoded, _, feature_names = preprocess_data(
    train_df.copy(),
    mode='train',
    imputer=imputer,
    scaler=scaler,
    protocol_encoder=protocol_label_encoder,
    attack_encoder=attack_label_encoder
)

# CatBoost를 위한 범주형 특성 인덱스 찾기
# 'protocol' 컬럼이 인코딩되었으므로, 해당 인덱스를 찾아 CatBoost에 전달
# 또는 protocol_label_encoder.transform하기 전에 protocol 컬럼의 인덱스를 저장
# 여기서는 protocol_label_encoder를 통과한 'protocol' 컬럼의 위치를 찾음
# X_train_processed는 numpy array이므로, 원래 컬럼명은 feature_names에서 찾아야 함
protocol_col_index = feature_names.get_loc('protocol') if 'protocol' in feature_names else -1

n_classes = len(np.unique(y_train_encoded))
print(f"총 클래스 수: {n_classes}")

### 3. 교차 검증 및 SMOTE 적용 (데이터 누수 방지) ###

N_SPLITS = 5 # K-Fold Cross-Validation 분할 수

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# 각 폴드에서의 모델 예측 확률 및 실제 라벨을 저장할 리스트
oof_preds_proba = np.zeros((X_train_processed.shape[0], n_classes))
oof_labels = np.zeros(X_train_processed.shape[0])

# 최종 테스트 세트 예측 확률을 위한 리스트
test_preds_proba_list = []

xgb_models = []
lgb_models = []
cat_models = []

print("\n" + "="*60)
print(f"K-Fold Cross-Validation ({N_SPLITS} Folds)")
print("="*60)
# 최종 테스트 데이터 전처리
X_test_scaled, _, test_ids, _ = preprocess_data(
    test_df.copy(),
    mode='test',
    imputer=imputer,
    scaler=scaler,
    protocol_encoder=protocol_label_encoder
)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y_train_encoded)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    X_train_fold, X_val_fold = X_train_processed[train_idx], X_train_processed[val_idx]
    y_train_fold, y_val_fold = y_train_encoded[train_idx], y_train_encoded[val_idx]

    # 폴드 내에서 SMOTE 적용
    original_distribution = Counter(y_train_fold)
    minority_classes = [class_id for class_id, count in original_distribution.items() if count <= 100]

    target_distribution = {}
    for class_id, count in original_distribution.items():
        if class_id in minority_classes and count <= 100:
            target_distribution[class_id] = min(count * 3, 300) # 최대 300개로 제한
        else:
            target_distribution[class_id] = count

    smote = SMOTE(sampling_strategy=target_distribution, random_state=42, k_neighbors=3)
    X_train_fold_balanced, y_train_fold_balanced = smote.fit_resample(X_train_fold, y_train_fold)

    print(f"  원본 학습 폴드 분포: {original_distribution}")
    print(f"  SMOTE 후 학습 폴드 분포: {Counter(y_train_fold_balanced)}")

    # 클래스 가중치 계산 (증강된 데이터 기준)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_fold_balanced),
        y=y_train_fold_balanced
    )
    class_weight_dict = dict(enumerate(class_weights))
    sample_weights = pd.Series(y_train_fold_balanced).map(class_weight_dict).values

    # 모델 학습
    print("  Training models...")

    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train_fold_balanced, y_train_fold_balanced, sample_weight=sample_weights)
    xgb_models.append(xgb_model)

    # LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective='multiclass',
        num_class=n_classes,
        metric='multi_logloss',
        random_state=42,
        verbosity=-1,
        force_col_wise=True # 데이터셋이 작아서 경고 나올 수 있음.
    )
    lgb_model.fit(X_train_fold_balanced, y_train_fold_balanced, sample_weight=sample_weights)
    lgb_models.append(lgb_model)

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        random_seed=42,
        verbose=False,
        # 'protocol' 컬럼이 Categorical임을 CatBoost에게 알려줍니다.
        # X_train_processed는 numpy array이므로, feature_names에서 인덱스를 찾아야 합니다.
        # 이 예시에서는 protocol_col_index를 사용합니다.
        #cat_features=[protocol_col_index] if protocol_col_index != -1 else []
    )
    cat_model.fit(X_train_fold_balanced, y_train_fold_balanced, sample_weight=sample_weights)
    cat_models.append(cat_model)

    # OOF (Out-Of-Fold) 예측 저장 (앙상블 스태킹에 활용 가능)
    # 각 모델의 예측 확률을 사용하여 평균 소프트 보팅 수행
    oof_xgb_proba = xgb_model.predict_proba(X_val_fold)
    oof_lgb_proba = lgb_model.predict_proba(X_val_fold)
    oof_cat_proba = cat_model.predict_proba(X_val_fold)

    oof_avg_proba = (oof_xgb_proba + oof_lgb_proba + oof_cat_proba) / 3
    oof_preds_proba[val_idx] = oof_avg_proba
    oof_labels[val_idx] = y_val_fold

    # 최종 테스트 세트에 대한 예측 (각 폴드 모델의 예측을 저장)
    test_preds_proba_list.append(
        (xgb_model.predict_proba(X_test_scaled) +
         lgb_model.predict_proba(X_test_scaled) +
         cat_model.predict_proba(X_test_scaled)) / 3
    )

# OOF 예측 성능 평가
oof_preds_final = np.argmax(oof_preds_proba, axis=1)
oof_accuracy = accuracy_score(oof_labels, oof_preds_final)
oof_f1_macro = f1_score(oof_labels, oof_preds_final, average='macro')
oof_f1_weighted = f1_score(oof_labels, oof_preds_final, average='weighted')

print("\n" + "="*60)
print("OOF (Out-Of-Fold) Ensemble Performance")
print("="*60)
print(f"Accuracy: {oof_accuracy:.4f}")
print(f"Macro F1: {oof_f1_macro:.4f}")
print(f"Weighted F1: {oof_f1_weighted:.4f}")


### 4. 최종 테스트 데이터 전처리 및 예측 ###

print("\n" + "="*60)
print("Final Prediction on Test Set")
print("="*60)

# # 최종 테스트 데이터 전처리
# X_test_scaled, _, test_ids, _ = preprocess_data(
#     test_df.copy(),
#     mode='test',
#     imputer=imputer,
#     scaler=scaler,
#     protocol_encoder=protocol_label_encoder
# )

# K-Fold 앙상블 (Soft Voting)을 통한 최종 예측
# test_preds_proba_list에 저장된 각 폴드 모델의 예측 확률을 평균
final_test_avg_proba = np.mean(test_preds_proba_list, axis=0)
final_test_preds_encoded = np.argmax(final_test_avg_proba, axis=1)

# 라벨 디코딩
final_test_pred_labels = attack_label_encoder.inverse_transform(final_test_preds_encoded)

# submission 파일 생성
submission_df = pd.DataFrame({'ID': test_ids, 'attack_type': final_test_pred_labels})
submission_df.to_csv(path+'optimized_submission.csv', index=False)

print("✅ Optimized submission file saved!")
print(f"Submission file shape: {submission_df.shape}")
print(submission_df.head())

# Accuracy: 0.9917
# Macro F1: 0.8104
# Weighted F1: 0.9919