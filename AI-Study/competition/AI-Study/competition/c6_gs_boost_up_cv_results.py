import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import QED
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
import joblib
import random
import datetime
import matplotlib.pyplot as plt

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
    return df

train = rdkit_features(train)
test = rdkit_features(test)

def prepare_data(df):
    feature_cols = [col for col in df.columns if col not in ['ID', 'Canonical_Smiles', 'Inhibition']]
    X = df[feature_cols]
    y = df['Inhibition'] if 'Inhibition' in df.columns else None
    return X, y, feature_cols

X, y, feature_cols = prepare_data(train)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

xgb_base = XGBRegressor(random_state=seed)
xgb_base.fit(X_train, y_train)
train_pred = xgb_base.predict(X_train)
val_pred = xgb_base.predict(X_val)
print('XGB - Train RMSE:', np.sqrt(mean_squared_error(y_train, train_pred)),
      'Val RMSE:', np.sqrt(mean_squared_error(y_val, val_pred)))
print('XGB - Train R2:', r2_score(y_train, train_pred),
      'Val R2:', r2_score(y_val, val_pred))

# feature importance 시각화 (gain 기준)
plt.figure(figsize=(12, 8))
plot_importance(xgb_base, importance_type='gain', max_num_features=20)
plt.title("XGBoost Feature Importances (gain)")
plt.tight_layout()
plt.show()

scaler = StandardScaler()
pca = PCA(n_components=0.95, random_state=seed)
pipeline = Pipeline([('scaler', scaler), ('pca', pca)])
X_train_pca = pipeline.fit_transform(X_train)
X_val_pca = pipeline.transform(X_val)
print(f"Original: {X_train.shape[1]} features -> PCA: {X_train_pca.shape[1]} components")

models = {
    'XGB': XGBRegressor(random_state=seed),
    'RF': RandomForestRegressor(random_state=seed),
    'Lasso': Lasso(random_state=seed),
    'SVR': SVR()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    yv = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, yv))
    r2 = r2_score(y_val, yv)
    model.fit(X_train_pca, y_train)
    yv_pca = model.predict(X_val_pca)
    rmse_pca = np.sqrt(mean_squared_error(y_val, yv_pca))
    r2_pca = r2_score(y_val, yv_pca)
    results.append({'Model': name,
                    'RMSE_orig': rmse, 'R2_orig': r2,
                    'RMSE_pca': rmse_pca, 'R2_pca': r2_pca})

results_df = pd.DataFrame(results).set_index('Model')
print(results_df)

best = 'XGB'
best_model = models[best]
best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
joblib.dump(best_model, path + 'best_model.pkl')
print('Best model saved: best_model.pkl')

X_test, _, _ = prepare_data(test)
test_preds = best_model.predict(X_test)
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
submission = pd.DataFrame({
    'ID': test['ID'],
    'Inhibition': test_preds
})
filename = f'{path}submission_final_{now}.csv'
submission.to_csv(filename, index=False)
print(f"제출 파일 저장 완료: {filename}")


# XGB - Train RMSE: 2.232204111123126 Val RMSE: 27.367898141558936
# XGB - Train R2: 0.9929409105945475 Val R2: -0.1336713318186311
# Original: 28 features -> PCA: 12 components
#        RMSE_orig   R2_orig   RMSE_pca    R2_pca
# Model
# XGB    27.367898 -0.133671  27.056378 -0.108010
# RF     24.166037  0.116076  24.282323  0.107548
# Lasso  23.717931  0.148553  23.951841  0.131675
# SVR    25.091963  0.047042  24.129509  0.118746
# Best model saved: best_model.pkl
# 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_final_20250710_161619.csv