import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs, MACCSkeys, AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import LinearRegression
import warnings
import multiprocessing
import optuna

warnings.filterwarnings('ignore')

SEED = 50
np.random.seed(SEED)
torch.manual_seed(SEED)

path = 'C:\Study25\_data\dacon\\boost_up\\'

# --- Morgan Fingerprint (512 bits) ---
def get_morgan_fp(mol, radius=2, n_bits=512):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

# --- ECFP4 Fingerprint (1024 bits) ---
def get_ecfp4_fp(mol, radius=2, n_bits=1024):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useFeatures=False)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

# --- FCFP4 Fingerprint (1024 bits) ---
def get_fcfp4_fp(mol, radius=2, n_bits=1024):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useFeatures=True)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

# --- MACCS Fingerprint (166 bits) ---
def get_maccs_fp(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

# --- Substructure matching ---
def get_substructure_features(mol):
    substructs = {
        'Amine': Chem.MolFromSmarts('[$([NX3H2]),$([NX3H][CX4]),$([NX3]([CX4])[CX4])]'),
        'CarboxylicAcid': Chem.MolFromSmarts('C(=O)[OH]'),
        'Amide': Chem.MolFromSmarts('NC(=O)'),
        'Imidazole': Chem.MolFromSmiles('c1cnc[nH]1'),
        'Pyridine': Chem.MolFromSmiles('c1ccncc1'),
        'Sulfonamide': Chem.MolFromSmarts('NS(=O)(=O)'),
        'Phenol': Chem.MolFromSmiles('c1ccccc1O'),
        'Thiol': Chem.MolFromSmiles('CS'),
        'Alcohol': Chem.MolFromSmarts('[CX4][OH]'),
        'Ketone': Chem.MolFromSmarts('C(=O)[#6]'),
        'Ether': Chem.MolFromSmarts('C-O-C'),
        'Thioether': Chem.MolFromSmarts('C-S-C'),
        'AromaticThioether': Chem.MolFromSmarts('c-S-c'),
        'Sulfonyl': Chem.MolFromSmarts('S(=O)(=O)'),
        'Phosphate': Chem.MolFromSmarts('P(=O)(O)(O)O'),
        'Aldehyde': Chem.MolFromSmarts('C(=O)[H]'),
        'Ester': Chem.MolFromSmarts('C(=O)O[#6]'),
        'Nitro': Chem.MolFromSmarts('N(=O)=O'),
        'Halogen': Chem.MolFromSmarts('[F,Cl,Br,I]')
    }
    return [int(mol.HasSubstructMatch(patt)) for patt in substructs.values()]

# --- Physicochemical descriptors ---
def get_physchem_features(mol):
    desc_list = [
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
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol),
        Descriptors.MaxAbsPartialCharge(mol),
        Descriptors.MinAbsPartialCharge(mol),
        Descriptors.Ipc(mol),
        Descriptors.Chi0n(mol),
        Descriptors.Chi1n(mol),
        Descriptors.Chi0v(mol),
        Descriptors.Chi1v(mol),
        Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol),
        Descriptors.Kappa3(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.PEOE_VSA1(mol),
        Descriptors.SMR_VSA1(mol),
        Descriptors.SlogP_VSA1(mol)
    ]
    return desc_list

# --- SMILES to feature ---
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        physchem = get_physchem_features(mol)
        morgan = get_morgan_fp(mol)
        ecfp4 = get_ecfp4_fp(mol)
        fcfp4 = get_fcfp4_fp(mol)
        maccs = get_maccs_fp(mol)
        substruct = get_substructure_features(mol)
        return physchem + morgan + ecfp4 + fcfp4 + maccs + substruct
    except Exception:
        return None

# --- 병렬 처리 ---
def extract_features_parallel(df, num_workers=multiprocessing.cpu_count()):
    with multiprocessing.Pool(processes=num_workers) as pool:
        features = pool.map(smiles_to_features, df['Canonical_Smiles'])
    return features

# --- Feature 준비 ---
def prepare_features(df, is_train=True, num_workers=multiprocessing.cpu_count()):
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

# --- 평가 함수: Normalized RMSE + Pearson + Score ---
def normalized_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    range_y = np.max(y_true) - np.min(y_true)
    if range_y == 0:
        return 0
    return rmse / range_y

def compute_score(y_true, y_pred):
    A = normalized_rmse(y_true, y_pred)
    B, _ = pearsonr(y_true, y_pred) if len(y_true) > 1 else (0, 0)
    score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    return A, B, score

# --- Optuna 튜닝: XGBoost ---
def tune_xgb(X, y, n_trials=20):
    def objective(trial):
        param = {
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist',  # GPU 사용
            'random_state': SEED,
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'n_estimators': 150
        }
        model = xgb.XGBRegressor(**param)
        model.fit(X, y)
        preds = model.predict(X)
        score = normalized_rmse(y, preds)
        return score
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# --- Optuna 튜닝: LightGBM ---
def tune_lgb(X, y, n_trials=20):
    def objective(trial):
        param = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'random_state': SEED,
            'verbosity': -1,
            'n_jobs': -1,
            'device': 'gpu',           # GPU 사용
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
            'bagging_freq': 5,
            'n_estimators': 150
        }
        model = lgb.LGBMRegressor(**param)
        model.fit(X, y)
        preds = model.predict(X)
        score = normalized_rmse(y, preds)
        return score
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def main():
    train = pd.read_csv(path + 'train.csv', index_col=0)
    test = pd.read_csv(path + 'test.csv', index_col=0)
    submission = pd.read_csv(path + 'sample_submission.csv')

    # 결측치 median으로 대체 (train 포함)
    train = train.fillna(train.median(numeric_only=True)).reset_index(drop=True)
    test = test.fillna(test.median(numeric_only=True))

    print("--- 특징 추출 시작 ---")
    X_train, y_train, train_idx = prepare_features(train, is_train=True)
    X_test, test_idx = prepare_features(test, is_train=False)
    print(f"학습 데이터 특징 shape: {X_train.shape}")
    print(f"테스트 데이터 특징 shape: {X_test.shape}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("--- 특징 스케일링 완료 ---")

    print("\n--- Optuna 튜닝 시작 (XGB, LGB) ---")
    best_xgb_params = tune_xgb(X_train, y_train, n_trials=20)
    best_lgb_params = tune_lgb(X_train, y_train, n_trials=20)

    print(f"Best XGB params: {best_xgb_params}")
    print(f"Best LGB params: {best_lgb_params}")

    # CatBoost 파라미터 (GPU)
    cat_params = {
        'iterations': 200,
        'depth': 6,
        'learning_rate': 0.05,
        'random_seed': SEED,
        'verbose': False,
        'task_type': 'GPU',
        'devices': '0'
    }

    model_xgb = xgb.XGBRegressor(**best_xgb_params, random_state=SEED, verbosity=0, n_jobs=-1, tree_method='gpu_hist')
    model_lgb = lgb.LGBMRegressor(**best_lgb_params, random_state=SEED, verbosity=-1, n_jobs=-1,
                                  device='gpu', gpu_platform_id=0, gpu_device_id=0)
    model_cat = cb.CatBoostRegressor(**cat_params)

    print("\n--- 모델 학습 시작 ---")
    model_xgb.fit(X_train, y_train)
    model_lgb.fit(X_train, y_train)
    model_cat.fit(X_train, y_train)

    print("\n--- 예측 및 앙상블 ---")
    preds_xgb = model_xgb.predict(X_test)
    preds_lgb = model_lgb.predict(X_test)
    preds_cat = model_cat.predict(X_test)

    preds_ensemble = (preds_xgb + preds_lgb + preds_cat) / 3

    submission.loc[test_idx, 'Inhibition'] = preds_ensemble
    submission.to_csv(path + 'submission_gpu_ensemble_optuna.csv', index=False)
    print("✅ 제출 파일 저장 완료: submission_gpu_ensemble_optuna.csv")

    # 학습 데이터에 대해 평가 결과 출력
    preds_train_ensemble = (model_xgb.predict(X_train) + model_lgb.predict(X_train) + model_cat.predict(X_train)) / 3
    A, B, score = compute_score(y_train, preds_train_ensemble)
    print(f"\n[학습 데이터 평가] Normalized RMSE: {A:.4f}, Pearson Correlation: {B:.4f}, Score: {score:.4f}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
