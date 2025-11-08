# train_xgboost.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from hybrid_extractor import HybridExtractor
import glob # for loading pretrained extractor
import warnings  # suppress warnings
warnings.filterwarnings('ignore')

# ── 재현성 위해 랜덤 시드 고정 ──
SEED = 404
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── 설정 디렉토리 ──
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR      = os.path.join(BASE_DIR, 'models')
SUBMISSION_DIR = os.path.join(BASE_DIR, 'submissions')
for d in (MODEL_DIR, SUBMISSION_DIR):
    os.makedirs(d, exist_ok=True)

# ── CLI 인자 정의 ──
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators',          type=int,   default=200)
parser.add_argument('--learning_rate',         type=float, default=0.1)
parser.add_argument('--max_depth',             type=int,   default=6)
parser.add_argument('--subsample',             type=float, default=0.8)
parser.add_argument('--colsample_bytree',      type=float, default=0.8)
parser.add_argument('--gamma',                 type=float, default=0.1)
parser.add_argument('--reg_alpha',             type=float, default=0.1)
parser.add_argument('--reg_lambda',            type=float, default=1.0)
parser.add_argument('--early_stopping_rounds', type=int,   default=20)
parser.add_argument('--tune',                  action='store_true')
args = parser.parse_args()

# ── 데이터 로더 (학습/검증) ──
def get_train_test_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    # 라벨 6을 1로 바꾸고 0 vs 1 이진화
    ds.targets[ds.targets == 6] = 1
    mask = (ds.targets == 0) | (ds.targets == 1)
    indices = torch.where(mask)[0]
    ds_bin = Subset(ds, indices)
    labels = [ds_bin[i][1] for i in range(len(ds_bin))]
    train_idx, val_idx = train_test_split(
        np.arange(len(ds_bin)), test_size=0.2,
        stratify=labels, random_state=42
    )
    train_ds = Subset(ds_bin, train_idx)
    val_ds   = Subset(ds_bin, val_idx)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    )

# ── 특징 추출 헬퍼 ──
def extract_features(loader, extractor, device):
    X_list, y_list = [], []
    extractor.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            feats = extractor(x_batch.to(device)).cpu().numpy()
            X_list.append(feats)
            y_list.append(y_batch.numpy())
    return np.vstack(X_list), np.concatenate(y_list)

# ── Optuna 목적 함수 ──
def objective(trial):
    params = {
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
    }
    es = trial.suggest_int('early_stopping_rounds', 5, 50)

    train_loader, val_loader = get_train_test_loaders(batch_size=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = HybridExtractor().to(device)
    X_tr, y_tr = extract_features(train_loader, extractor, device)
    X_val, y_val = extract_features(val_loader, extractor, device)

    # 교차 검증으로 과적합 방지
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    model_cv = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model_cv, X_tr, y_tr, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# ── 하이퍼파라미터 튜닝 ──
if args.tune:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    best = study.best_params
    args.n_estimators           = best['n_estimators']
    args.learning_rate          = best['learning_rate']
    args.max_depth              = best['max_depth']
    args.subsample              = best['subsample']
    args.colsample_bytree       = best['colsample_bytree']
    args.early_stopping_rounds  = best['early_stopping_rounds']

# ── 최종 학습 및 제출 ──
# 1) 학습/검증 세트 특징 추출
train_loader, val_loader = get_train_test_loaders(batch_size=64)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
submission_ds     = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform_test
)
submission_loader = DataLoader(submission_ds, batch_size=64, shuffle=False)

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor = HybridExtractor().to(device)
# Load pretrained extractor: select latest extractor_*.pt in models/
extractor = HybridExtractor().to(device)
extractor_paths = glob.glob(os.path.join(MODEL_DIR, 'extractor_*.pt'))
if not extractor_paths:
    raise FileNotFoundError(f"No pretrained extractor found in {MODEL_DIR}/")
latest_ext = max(extractor_paths, key=os.path.getmtime)
extractor.load_state_dict(torch.load(latest_ext))
extractor.eval()

# 특징 추출
X_tr, y_tr   = extract_features(train_loader, extractor, device)
X_val, y_val = extract_features(val_loader, extractor, device)
X_sub, _     = extract_features(submission_loader, extractor, device)

# 2) XGBoost 학습
xgb_params = {
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'n_estimators': args.n_estimators,
    'learning_rate': args.learning_rate,
    'max_depth': args.max_depth,
    'subsample': args.subsample,
    'colsample_bytree': args.colsample_bytree,
    'gamma': args.gamma,
    'reg_alpha': args.reg_alpha,
    'reg_lambda': args.reg_lambda
}
model = XGBClassifier(**xgb_params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=args.early_stopping_rounds,
    verbose=10
)

# 3) 검증 지표 출력
pred_val = model.predict(X_val)
pred_val = np.where(pred_val == 1, 6, pred_val)
y_val_orig = np.where(y_val == 1, 6, y_val)
print("Validation metrics:")
print(f"Accuracy : {accuracy_score(y_val_orig, pred_val):.4f}")
print(f"Precision: {precision_score(y_val_orig, pred_val, pos_label=6):.4f}")
print(f"Recall   : {recall_score(y_val_orig, pred_val, pos_label=6):.4f}")
print(f"F1 Score : {f1_score(y_val_orig, pred_val, pos_label=6):.4f}")

# 4) 모델 저장
# Timestamp for filenames
ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
mod_fp = os.path.join(MODEL_DIR, f'xgb_{ts}.json')
model.save_model(mod_fp)
print(f'XGBoost model saved: {mod_fp}')

# 5) 제출 예측 및 저장
preds = model.predict(X_sub)
preds = np.where(preds == 1, 6, preds)
assert len(preds) == len(submission_ds)
sub_fp = os.path.join(SUBMISSION_DIR, f'y_pred_{ts}.csv')
pd.DataFrame(preds).to_csv(sub_fp, index=False, header=False)
print(f'Submission saved: {sub_fp}')