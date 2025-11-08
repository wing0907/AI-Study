import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit import RDLogger
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import joblib
import random
import datetime

RDLogger.DisableLog('rdApp.*')

# 설정
seed = 222
random.seed(seed)
np.random.seed(seed)

# 경로 설정
path = 'C:/Study25/_data/dacon/boost_up/'
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
test_ids = test['ID'].copy()

# SMARTS 기반 기능성 그룹

def rdkit_features(df):
    phenol_group = Chem.MolFromSmarts("c[OH]")
    alcohol_group = Chem.MolFromSmarts("[CX4][OH]")
    ketone_group = Chem.MolFromSmarts("[CX3](=O)[#6]")
    thiol_ether = Chem.MolFromSmarts("[#6]-S-[#6]")
    aromatic_thiol_ether = Chem.MolFromSmarts("c-S-[#6]")
    sulfonyl_group = Chem.MolFromSmarts("S(=O)(=O)")
    phosphates_group = Chem.MolFromSmiles("P(=O)")
    amide_group = Chem.MolFromSmarts("[NX3][CX3](=O)[#6]")

    mols = df['Canonical_Smiles'].apply(Chem.MolFromSmiles)
    df = df.copy()
    df['Mol'] = mols
    df['MolLogP'] = mols.apply(Crippen.MolLogP)
    df['NumHAcceptors'] = mols.apply(Lipinski.NumHAcceptors)
    df['NumHDonors'] = mols.apply(Lipinski.NumHDonors)
    df['TPSA'] = mols.apply(Descriptors.TPSA)
    df['NumAromaticRings'] = mols.apply(Lipinski.NumAromaticRings)
    df['NumAromaticHeterocycles'] = mols.apply(Lipinski.NumAromaticHeterocycles)
    df['NumSaturatedHeterocycles'] = mols.apply(Lipinski.NumSaturatedHeterocycles)
    df['NOCount'] = mols.apply(Descriptors.NOCount)
    df['NumRadicalElectrons'] = mols.apply(Descriptors.NumRadicalElectrons)
    df['NHOHCount'] = mols.apply(Descriptors.NHOHCount)
    df['HasPhenol'] = mols.apply(lambda m: m.HasSubstructMatch(phenol_group)).astype(int)
    df['HasAlcohol'] = mols.apply(lambda m: m.HasSubstructMatch(alcohol_group)).astype(int)
    df['HasKetone'] = mols.apply(lambda m: m.HasSubstructMatch(ketone_group)).astype(int)
    df['HasThioether'] = mols.apply(lambda m: m.HasSubstructMatch(thiol_ether)).astype(int)
    df['HasArylThioether'] = mols.apply(lambda m: m.HasSubstructMatch(aromatic_thiol_ether)).astype(int)
    df['HasSulfonyl'] = mols.apply(lambda m: m.HasSubstructMatch(sulfonyl_group)).astype(int)
    df['HasPhosphate'] = mols.apply(lambda m: m.HasSubstructMatch(phosphates_group)).astype(int)
    df['HasAmide'] = mols.apply(lambda m: m.HasSubstructMatch(amide_group)).astype(int)
    return df

train = rdkit_features(train)
test = rdkit_features(test)

feature_cols = [col for col in train.columns if col not in ['ID', 'Canonical_Smiles', 'Inhibition', 'Mol']]
X = train[feature_cols]
y = train['Inhibition']
test_X = test[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_X)

kf = KFold(n_splits=5, shuffle=True, random_state=seed)
r2s, rmses = [], []
best_model = None
best_r2 = -np.inf

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    print(f"Fold {fold+1}")
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgbm = LGBMRegressor(
        random_state=seed,
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.03,
        objective='regression',
        metric='rmse',
        verbosity=-1,
        force_col_wise=True
    )
    lgbm.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             eval_metric='rmse',
             callbacks=[lgb_early_stopping(stopping_rounds=30, verbose=False)])

    cat = CatBoostRegressor(
        verbose=0,
        random_state=seed,
        iterations=1000,
        depth=4,
        learning_rate=0.03,
        loss_function='RMSE'
    )
    cat.fit(X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=30)

    gbr = GradientBoostingRegressor(random_state=seed,
                                     n_estimators=500,
                                     max_depth=3,
                                     learning_rate=0.03)
    gbr.fit(X_train, y_train)

    rfr = RandomForestRegressor(random_state=seed,
                                 n_estimators=300,
                                 max_depth=6,
                                 n_jobs=-1)
    rfr.fit(X_train, y_train)

    stacking_model = StackingRegressor(
        estimators=[
            ('lgbm', lgbm),
            ('cat', cat),
            ('gbr', gbr),
            ('rfr', rfr)
        ],
        final_estimator=GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05),
        cv=3,
        passthrough=True
    )

    stacking_model.fit(X_train, y_train)
    preds = stacking_model.predict(X_val)
    r2 = r2_score(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2s.append(r2)
    rmses.append(rmse)
    print(f"  R2: {r2:.4f}, RMSE: {rmse:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = stacking_model

print(f"\nBest R2 from folds: {best_r2:.4f}")
print(f"Average R2: {np.mean(r2s):.4f}")
print(f"Average RMSE: {np.mean(rmses):.4f}")

test_preds = best_model.predict(test_scaled)

submission = pd.DataFrame({
    'ID': test_ids,
    'Inhibition': test_preds
})

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = path + f'best_stacking_model_{now}.pkl'
submission_path = path + f'submission_best_stacking_{now}.csv'

joblib.dump(best_model, model_path)
joblib.dump(scaler, path + f'scaler_{now}.pkl')
submission.to_csv(submission_path, index=False)

print(f"\n✔️ 최고 R² 모델 저장 완료: {model_path}")
print(f"📄 제출 파일 저장 완료: {submission_path}")


# kfold=5
# Average R2: -0.2790
# Average RMSE: 29.7913
# 제출 파일 저장 완료: voting_submission_20250717_170854.csv

# kfold=8
# Average R2: -0.2777
# Average RMSE: 29.7267
# 제출 파일 저장 완료: voting_submission_20250717_171245.csv


# Average R2: -0.1584
# Average RMSE: 28.3555
# 제출 파일 저장 완료: voting_submission_20250717_172850.csv


# kfold=5 800/500/300
# Average R2: 0.0924
# Average RMSE: 25.1067
# 제출 파일 저장 완료: voting_submission_20250717_173502.csv


# Average R2: 0.0355
# Average RMSE: 25.8783
# 제출 파일 저장 완료: voting_submission_20250717_174840.csv

# 스태킹 + 앙상블
# Average R2: 0.0697
# Average RMSE: 25.4129
# 모델 저장 완료: C:/Study25/_data/dacon/boost_up/stacking_model_20250717_180703.pkl
# 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_stacking_20250717_180703.csv


# Best R2 from folds: 0.1268
# Average R2: 0.1043
# Average RMSE: 24.9397
# ✔️ 최고 R² 모델 저장 완료: C:/Study25/_data/dacon/boost_up/best_stacking_model_20250717_182251.pkl
# 📄 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_best_stacking_20250717_182251.csv