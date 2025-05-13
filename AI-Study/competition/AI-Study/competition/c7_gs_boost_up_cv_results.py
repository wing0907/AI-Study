import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
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
import warnings
warnings.filterwarnings('ignore')

# 0. 설정
seed = 222
random.seed(seed)
np.random.seed(seed)

# 1. 데이터 로드 및 RDKit 파생변수 생성
path = 'C:/Study25/_data/dacon/boost_up/'
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

def rdkit_features(df):
    mols = df['Canonical_Smiles'].apply(Chem.MolFromSmiles)
    df = df.copy()
    df['MolWt'] = mols.apply(Descriptors.MolWt)
    df['MolLogP'] = mols.apply(Crippen.MolLogP)
    df['NumHAcceptors'] = mols.apply(Lipinski.NumHAcceptors)
    df['NumHDonors'] = mols.apply(Lipinski.NumHDonors)
    df['TPSA'] = mols.apply(Descriptors.TPSA)
    df['NumRotatableBonds'] = mols.apply(Descriptors.NumRotatableBonds)
    df['NumAromaticRings'] = mols.apply(Lipinski.NumAromaticRings)
    df['NumHeteroatoms'] = mols.apply(Descriptors.NumHeteroatoms)
    df['FractionCSP3'] = mols.apply(Descriptors.FractionCSP3)
    df['NumAliphaticRings'] = mols.apply(Lipinski.NumAliphaticRings)
    df['NumAromaticHeterocycles'] = mols.apply(Lipinski.NumAromaticHeterocycles)
    df['NumSaturatedHeterocycles'] = mols.apply(Lipinski.NumSaturatedHeterocycles)
    df['NumAliphaticHeterocycles'] = mols.apply(Lipinski.NumAliphaticHeterocycles)
    df['HeavyAtomCount'] = mols.apply(Descriptors.HeavyAtomCount)
    df['RingCount'] = mols.apply(Descriptors.RingCount)
    df['NOCount'] = mols.apply(Descriptors.NOCount)
    df['NHOHCount'] = mols.apply(Descriptors.NHOHCount)
    df['NumRadicalElectrons'] = mols.apply(Descriptors.NumRadicalElectrons)
    df['ExactMolWt'] = mols.apply(Descriptors.ExactMolWt)
    df['NumValenceElectrons'] = mols.apply(Descriptors.NumValenceElectrons)
    df['NumAmideBonds'] = mols.apply(rdMolDescriptors.CalcNumAmideBonds)
    df['NumSaturatedCarbocycles'] = mols.apply(rdMolDescriptors.CalcNumSaturatedCarbocycles)
    df['NumSaturatedRings'] = mols.apply(rdMolDescriptors.CalcNumSaturatedRings)
    df['NumSpiroAtoms'] = mols.apply(rdMolDescriptors.CalcNumSpiroAtoms)
    df['NumBridgeheadAtoms'] = mols.apply(rdMolDescriptors.CalcNumBridgeheadAtoms)
    df['LabuteASA'] = mols.apply(rdMolDescriptors.CalcLabuteASA)
    df['BalabanJ'] = mols.apply(Descriptors.BalabanJ)
    df['BertzCT'] = mols.apply(Descriptors.BertzCT)

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

train = rdkit_features(train)
test = rdkit_features(test)

feature_cols = [col for col in train.columns if col not in ['ID', 'Canonical_Smiles', 'Inhibition']]
X = train[feature_cols]
y = train['Inhibition']

print(f"전체 피처 수: {len(feature_cols)}")

xgb_model = XGBRegressor(random_state=seed)
xgb_model.fit(X, y)

booster = xgb_model.get_booster()
importances = booster.get_score(importance_type='gain')
importance_df = pd.DataFrame({
    'Feature': list(importances.keys()),
    'Gain': list(importances.values())
}).sort_values(by='Gain', ascending=False).reset_index(drop=True)

# 하위 25% 피처 제거
threshold_idx = int(len(importance_df) * 0.25)
low_importance_features = importance_df.iloc[-threshold_idx:]['Feature'].tolist()
X_filtered = X.drop(columns=low_importance_features)

# 앙상블 학습
xgb_final = XGBRegressor(random_state=seed)
rf_final = RandomForestRegressor(random_state=seed)
gbr_final = GradientBoostingRegressor(random_state=seed)
ensemble_model = VotingRegressor(estimators=[
    ('xgb', xgb_final),
    ('rf', rf_final),
    ('gbr', gbr_final)
])
ensemble_model.fit(X_filtered, y)

# 성능 평가
train_pred = ensemble_model.predict(X_filtered)
rmse = np.sqrt(mean_squared_error(y, train_pred))
r2 = r2_score(y, train_pred)
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# 평가 산식
max_inh = y.max()
min_inh = y.min()
correlation = np.corrcoef(y, train_pred)[0, 1]
A = rmse / (max_inh - min_inh)
B = correlation
score = 0.5 * (1 - min(A, 1)) + 0.5 * B
print(f"Max Inhibition: {max_inh:.4f}")
print(f"Min Inhibition: {min_inh:.4f}")
print(f"선형 상관계수: {correlation:.4f}")
print(f"Score: {score:.4f}")

# 모델 저장
joblib.dump(ensemble_model, path + 'final_ensemble_model.pkl')
print("모델 저장 완료: final_ensemble_model.pkl")

# Feature importance 재확인
plt.figure(figsize=(12, 8))
plot_importance(xgb_model, importance_type='gain', max_num_features=20)
plt.title("XGBoost Feature Importances (gain)")
plt.tight_layout()
plt.show()

# 테스트 데이터 준비 및 예측
X_test = test[X_filtered.columns]
preds = ensemble_model.predict(X_test)

# 제출 파일 생성
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
submission = pd.DataFrame({
    'ID': test['ID'],
    'Inhibition': preds
})
filename = f'{path}submission_ensemble_{now}.csv'
submission.to_csv(filename, index=False)
print(f"제출 파일 저장 완료: {filename}")


# RMSE: 10.0150
# R² Score: 0.8561
# Max Inhibition: 99.3815
# Min Inhibition: 0.0000
# 선형 상관계수: 0.9677
# Score: 0.9335
# 모델 저장 완료: final_ensemble_model.pkl
# 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_ensemble_20250710_175944.csv