import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
import multiprocessing

warnings.filterwarnings('ignore')

SEED = 222
np.random.seed(SEED)
torch.manual_seed(SEED)

path = './_data/dacon/Drug/'

# --- Morgan Fingerprint (512 bits로 변경) ---
def get_morgan_fp(mol, radius=2, n_bits=512):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# --- 화학적 피처 추출 ---
def get_physchem_features(mol):
    return [
        Descriptors.MolWt(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.HeavyAtomMolWt(mol),
        Descriptors.MolMR(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NOCount(mol),
        Descriptors.NHOHCount(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumRadicalElectrons(mol),
        Descriptors.NumAromaticHeterocycles(mol),
        Descriptors.NumAliphaticHeterocycles(mol),
        Descriptors.NumSaturatedHeterocycles(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        Descriptors.FpDensityMorgan1(mol),
        Descriptors.FpDensityMorgan2(mol),
        Descriptors.FpDensityMorgan3(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Descriptors.BertzCT(mol),
        Descriptors.HallKierAlpha(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol),
        Descriptors.MaxAbsPartialCharge(mol),
        Descriptors.MinAbsPartialCharge(mol),
        Descriptors.Ipc(mol),
    ]

# --- SMILES to Feature ---
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        physchem = get_physchem_features(mol)
        morgan = get_morgan_fp(mol)
        return physchem + list(morgan)
    except:
        return None

# --- 병렬 처리 ---
def extract_features_parallel(df, num_workers=4):
    pool = multiprocessing.Pool(processes=num_workers)
    features = pool.map(smiles_to_features, df['Canonical_Smiles'])
    pool.close()
    pool.join()
    return features

def prepare_features(df, is_train=True, num_workers=4):
    features = extract_features_parallel(df, num_workers)
    filtered_features = []
    targets = []
    indices = []
    for idx, feat in enumerate(features):
        if feat is None:
            continue
        filtered_features.append(feat)
        indices.append(idx)
        if is_train:
            targets.append(df.iloc[idx]['Inhibition'])
    filtered_features = np.array(filtered_features)
    if is_train:
        return filtered_features, np.array(targets), indices
    return filtered_features, indices

# --- 커스텀 평가 지표 함수 ---
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    range_y = np.max(y_true) - np.min(y_true)
    normalized_rmse = rmse / range_y if range_y != 0 else 0
    normalized_rmse = min(normalized_rmse, 1)
    corr, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 0)
    score = 0.5 * (1 - normalized_rmse) + 0.5 * corr
    return normalized_rmse, corr, score

# --- XGBoost Optuna 목적함수 ---
def xgb_objective(trial, X, y):
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e-1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e-1, log=True),
        'random_state': SEED,
        'verbosity': 0,
    }
    model = xgb.XGBRegressor(**params)
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[val_idx])
        _, _, sc = compute_metrics(y[val_idx], pred)
        scores.append(sc)
    return -np.mean(scores)

# --- LightGBM Optuna 목적함수 ---
def lgb_objective(trial, X, y):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 1e-1, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 1e-1, log=True),
        'seed': SEED
    }
    model = lgb.LGBMRegressor(**params)
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[val_idx])
        _, _, sc = compute_metrics(y[val_idx], pred)
        scores.append(sc)
    return -np.mean(scores)

# --- CatBoost Optuna 목적함수 ---
def cat_objective(trial, X, y):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 600),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': False,
        'random_seed': SEED
    }
    model = cb.CatBoostRegressor(**params)
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[val_idx])
        _, _, sc = compute_metrics(y[val_idx], pred)
        scores.append(sc)
    return -np.mean(scores)

def main():
    train = pd.read_csv(path + 'train.csv', index_col=0)
    test = pd.read_csv(path + 'test.csv', index_col=0)
    submission = pd.read_csv(path + 'sample_submission.csv')

    train = train.dropna().reset_index(drop=True)
    test = test.fillna(test.median(numeric_only=True))

    X_train, y_train, train_idx = prepare_features(train, is_train=True)
    X_test, test_idx = prepare_features(test, is_train=False)

    # PCA 제거: 차원 축소하지 않고 스케일링만 수행
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[Optuna] XGBoost 튜닝 시작...")
    study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_xgb.optimize(lambda trial: xgb_objective(trial, X_train, y_train), n_trials=50)
    print("[Optuna] XGBoost 최적 파라미터:", study_xgb.best_params)

    print("[Optuna] LightGBM 튜닝 시작...")
    study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_lgb.optimize(lambda trial: lgb_objective(trial, X_train, y_train), n_trials=50)
    print("[Optuna] LightGBM 최적 파라미터:", study_lgb.best_params)

    print("[Optuna] CatBoost 튜닝 시작...")
    study_cat = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_cat.optimize(lambda trial: cat_objective(trial, X_train, y_train), n_trials=50)
    print("[Optuna] CatBoost 최적 파라미터:", study_cat.best_params)

    model_xgb = xgb.XGBRegressor(**study_xgb.best_params, random_state=SEED, verbosity=0)
    model_xgb.fit(X_train, y_train)
    model_lgb = lgb.LGBMRegressor(**study_lgb.best_params, random_state=SEED)
    model_lgb.fit(X_train, y_train)
    model_cat = cb.CatBoostRegressor(**study_cat.best_params, random_seed=SEED, verbose=False)
    model_cat.fit(X_train, y_train)

    preds_xgb = model_xgb.predict(X_test)
    preds_lgb = model_lgb.predict(X_test)
    preds_cat = model_cat.predict(X_test)

    # 각 모델별 CV 점수를 기반으로 가중치 계산 (가중 평균)
    w_xgb = max(study_xgb.best_value, 0)
    w_lgb = max(study_lgb.best_value, 0)
    w_cat = max(study_cat.best_value, 0)
    total_w = w_xgb + w_lgb + w_cat
    w_xgb = (total_w - w_xgb) / total_w
    w_lgb = (total_w - w_lgb) / total_w
    w_cat = (total_w - w_cat) / total_w

    preds_ensemble = (preds_xgb * w_xgb + preds_lgb * w_lgb + preds_cat * w_cat) / (w_xgb + w_lgb + w_cat)

    submission.loc[test_idx, 'Inhibition'] = preds_ensemble
    submission.to_csv(path + 'submission_final_ensemble_improved.csv', index=False)
    print("✅ 제출 파일 저장 완료: submission_final_ensemble_improved.csv")

    # 학습 데이터에 대한 최종 평가 출력
    preds_train_xgb = model_xgb.predict(X_train)
    preds_train_lgb = model_lgb.predict(X_train)
    preds_train_cat = model_cat.predict(X_train)
    preds_train_ensemble = (preds_train_xgb * w_xgb + preds_train_lgb * w_lgb + preds_train_cat * w_cat) / (w_xgb + w_lgb + w_cat)

    a, b, sc = compute_metrics(y_train, preds_train_ensemble)
    print(f"\n[최종 학습 데이터 평가] Normalized RMSE: {a:.4f}, Pearson: {b:.4f}, Combined Score: {sc:.4f}\n")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
