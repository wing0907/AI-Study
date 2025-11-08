import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
import joblib
import random
import datetime
import matplotlib.pyplot as plt
import os
# 경고 메시지 제거
RDLogger.DisableLog('rdApp.*')

# 0. 설정
seed = 222
random.seed(seed)
np.random.seed(seed)

# 1. 데이터 로드 및 RDKit 파생변수 생성
path = 'C:/Study25/_data/dacon/boost_up/'
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

print(os.getcwd())

def rdkit_features(df):
    mols = df['Canonical_Smiles'].apply(Chem.MolFromSmiles)
    df = df.copy()
       # 중요 피처들 (별표 있는 항목은 weight 조절 가능하도록 별도 저장)
    df['MolLogP'] = mols.apply(Crippen.MolLogP)  # *
    df['NumHAcceptors'] = mols.apply(Lipinski.NumHAcceptors)
    df['NumHDonors'] = mols.apply(Lipinski.NumHDonors)  # **
    df['TPSA'] = mols.apply(Descriptors.TPSA)  # **
    df['NumAromaticRings'] = mols.apply(Lipinski.NumAromaticRings)  # **
    df['FractionCSP3'] = mols.apply(Descriptors.FractionCSP3)  # **
    df['NumAromaticHeterocycles'] = mols.apply(Lipinski.NumAromaticHeterocycles)
    df['NumSaturatedHeterocycles'] = mols.apply(Lipinski.NumSaturatedHeterocycles)
    df['NOCount'] = mols.apply(Descriptors.NOCount)  # **
    df['NumAmideBonds'] = mols.apply(rdMolDescriptors.CalcNumAmideBonds)  # **
    df['NumRadicalElectrons'] = mols.apply(Descriptors.NumRadicalElectrons)
    df['NHOHCount'] = mols.apply(Descriptors.NHOHCount)

    # 서브구조 매칭 피처들
    phenol_group = Chem.MolFromSmiles("c1ccccc1O")
    alcohol_group = Chem.MolFromSmiles("CO")
    ketone_group = Chem.MolFromSmiles("C=O")
    thiol_ether = Chem.MolFromSmiles("CS")
    aromatic_thiol_ether = Chem.MolFromSmiles("c1ccccc1S")
    sulfonyl_group = Chem.MolFromSmiles("S(=O)(=O)")
    phosphates_group = Chem.MolFromSmiles("P(=O)")

    df['PhenolGroup'] = mols.apply(lambda mol: mol.HasSubstructMatch(phenol_group))
    df['AlcoholGroup'] = mols.apply(lambda mol: mol.HasSubstructMatch(alcohol_group))
    df['KetoneGroup'] = mols.apply(lambda mol: mol.HasSubstructMatch(ketone_group))
    df['ThiolGroup'] = mols.apply(lambda mol: mol.HasSubstructMatch(thiol_ether))
    df['AromaticThiolGroup'] = mols.apply(lambda mol: mol.HasSubstructMatch(aromatic_thiol_ether))
    df['SulfonylGroup'] = mols.apply(lambda mol: mol.HasSubstructMatch(sulfonyl_group))
    df['PhosphatesGroup'] = mols.apply(lambda mol: mol.HasSubstructMatch(phosphates_group))


    def calc_ecfp_avg(mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros((1,), dtype=np.int8)
        import rdkit.DataStructs.cDataStructs as DataStructs
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    ecfps = mols.apply(calc_ecfp_avg)
    ecfp_array = np.array(ecfps.tolist())
    ecfp_df = pd.DataFrame(ecfp_array, columns=[f'ECFP_{i}' for i in range(ecfp_array.shape[1])])
    df = pd.concat([df.reset_index(drop=True), ecfp_df], axis=1)

    return df



##########################################################################################
from rdkit.Chem import Descriptors, rdMolDescriptors

def calculate_all_rdkit_descriptors(mol):
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    if mol is None:
        return dict.fromkeys([
            *descriptor_names
        ], np.nan)   
    calc = MolecularDescriptorCalculator(descriptor_names)
    results = dict(zip(descriptor_names, calc.CalcDescriptors(mol)))
    return pd.Series(results)

# df = pd.read_csv(path + 'train.csv')
df = pd.read_csv('C:/Study25/_data/dacon/boost_up/train.csv')
smiles = df['Canonical_Smiles'].apply(lambda x:Chem.MolFromSmiles(x))
features = smiles.apply(calculate_all_rdkit_descriptors)
df = pd.concat([df, features], axis=1)
##########################################################################################

train = rdkit_features(train)
test = rdkit_features(test)

feature_cols = [col for col in train.columns if col not in ['ID', 'Canonical_Smiles', 'Inhibition']]
X = train[feature_cols]
y = train['Inhibition']

# 로그 변환
log_y = np.log1p(y)

print(f"전체 피처 수: {len(feature_cols)}")

# 퍼센티지별 성능 저장
results = []

for pct in [0.05, 0.10, 0.15, 0.20, 0.25]:
    xgb_model = XGBRegressor(random_state=seed)
    xgb_model.fit(X, log_y)
    booster = xgb_model.get_booster()
    importances = booster.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': list(importances.keys()),
        'Gain': list(importances.values())
    })
    importance_df['Gain'] = importance_df['Gain'] / importance_df['Gain'].sum()  # 정규화된 gain 값

    threshold_idx = int(len(importance_df) * pct)
    low_importance_features = importance_df.nsmallest(threshold_idx, 'Gain')['Feature'].tolist()
    # print("삭제된 컬럼들:", low_importance_features)

    x_f = X.drop(columns=low_importance_features)
    kf = KFold(n_splits=8, shuffle=True, random_state=seed)
    preds = np.zeros(len(x_f))

    for train_idx, val_idx in kf.split(x_f):
        X_train, X_val = x_f.iloc[train_idx], x_f.iloc[val_idx]
        y_train, y_val = log_y.iloc[train_idx], log_y.iloc[val_idx]

        model = VotingRegressor(estimators=[
            ('xgb', XGBRegressor(random_state=seed)),
            ('rf', RandomForestRegressor(random_state=seed)),
            ('gbr', GradientBoostingRegressor(random_state=seed))
        ])
        model.fit(X_train, y_train)
        preds[val_idx] = model.predict(X_val)

    final_preds = np.expm1(preds)
    rmse = np.sqrt(mean_squared_error(y, final_preds))
    r2 = r2_score(y, final_preds)
    correlation = np.corrcoef(y, final_preds)[0, 1]
    A = rmse / (y.max() - y.min())
    B = correlation
    score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    print(f"[{int(pct * 100)}% 제거] RMSE: {rmse:.4f}, R²: {r2:.4f}, Corr: {correlation:.4f}, Score: {score:.4f}")

    results.append((pct, score, rmse, r2, correlation, x_f.columns.tolist(), model))

# 최고 성능 퍼센티지 선택
best_pct, best_score, best_rmse, best_r2, best_corr, best_features, best_model = max(results, key=lambda x: x[1])

print('SEED :', seed)
print(f"\n최종 선택된 퍼센트: {int(best_pct * 100)}%, Score: {best_score:.4f}")

# 최종 모델 저장 및 제출 파일 생성
X_test = test[best_features]
test_preds_log = best_model.predict(X_test)
test_preds = np.expm1(test_preds_log)

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
submission = pd.DataFrame({
    'ID': test['ID'],
    'Inhibition': test_preds
})
filename = f'{path}submission_best_{now}.csv'
submission.to_csv(filename, index=False)
print(f"제출 파일 저장 완료: {filename}")

# 모델 저장
joblib.dump(best_model, path + 'final_best_model.pkl')
print("모델 저장 완료: final_best_model.pkl")


# 전체 피처 수: 1052 //seed 222
# [5% 제거] RMSE: 27.3161, R²: -0.0708, Corr: 0.3668, Score: 0.5460
# [10% 제거] RMSE: 27.1972, R²: -0.0615, Corr: 0.3736, Score: 0.5500
# [15% 제거] RMSE: 27.1718, R²: -0.0595, Corr: 0.3790, Score: 0.5528
# [20% 제거] RMSE: 27.0247, R²: -0.0480, Corr: 0.3849, Score: 0.5565
# [25% 제거] RMSE: 27.2976, R²: -0.0693, Corr: 0.3664, Score: 0.5459

# 최종 선택된 퍼센트: 20%, Score: 0.5565
# 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_best_20250710_192914.csv
# 모델 저장 완료: final_best_model.pkl




# [5% 제거] RMSE: 27.4978, R²: -0.0851, Corr: 0.3503, Score: 0.5368
# [10% 제거] RMSE: 27.5345, R²: -0.0880, Corr: 0.3475, Score: 0.5352
# [15% 제거] RMSE: 27.6170, R²: -0.0945, Corr: 0.3366, Score: 0.5293
# [20% 제거] RMSE: 27.5525, R²: -0.0894, Corr: 0.3509, Score: 0.5368
# [25% 제거] RMSE: 27.4800, R²: -0.0837, Corr: 0.3555, Score: 0.5395
# SEED : 7984

# 최종 선택된 퍼센트: 25%, Score: 0.5395
# 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_best_20250717_163902.csv
# 모델 저장 완료: final_best_model.pkl


