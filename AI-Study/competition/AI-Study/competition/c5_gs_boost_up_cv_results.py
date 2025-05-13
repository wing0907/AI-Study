import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
import joblib
import random
import datetime

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
    # 기본 물성치
    df['MolWt'] = mols.apply(Descriptors.MolWt)
    df['TPSA'] = mols.apply(Descriptors.TPSA)
    df['MolLogP'] = mols.apply(Descriptors.MolLogP)
    df['NumRotatableBonds'] = mols.apply(Descriptors.NumRotatableBonds)
    df['NumHAcceptors'] = mols.apply(Descriptors.NumHAcceptors)
    df['NumHDonors'] = mols.apply(Descriptors.NumHDonors)
    # 추가 물성치
    df['MolMR'] = mols.apply(Descriptors.MolMR)
    df['HeavyAtomCount'] = mols.apply(Descriptors.HeavyAtomCount)
    df['RingCount'] = mols.apply(Descriptors.RingCount)
    df['NumAromaticRings'] = mols.apply(Descriptors.NumAromaticRings)
    df['FractionCSP3'] = mols.apply(Descriptors.FractionCSP3)
    df['BertzCT'] = mols.apply(Descriptors.BertzCT)
    return df

# 파생변수 적용
train = rdkit_features(train)
test = rdkit_features(test)

def prepare_data(df):
    feature_cols = [
        'MolWt','TPSA','MolLogP','MolMR','HeavyAtomCount','RingCount',
        'NumRotatableBonds','NumHAcceptors','NumHDonors',
        'NumAromaticRings','FractionCSP3','BertzCT'
    ]
    X = df[feature_cols]
    y = df['Inhibition'] if 'Inhibition' in df.columns else None
    return X, y, feature_cols

# 2. 훈련/검증 분리
X, y, feature_cols = prepare_data(train)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

# 3. 과적합 확인: XGB 기본 모델
xgb_base = XGBRegressor(random_state=seed)
xgb_base.fit(X_train, y_train)
train_pred = xgb_base.predict(X_train)
val_pred = xgb_base.predict(X_val)
print('XGB - Train RMSE:', mean_squared_error(y_train, train_pred, squared=False),
      'Val RMSE:', mean_squared_error(y_val, val_pred, squared=False))
print('XGB - Train R2:', r2_score(y_train, train_pred),
      'Val R2:', r2_score(y_val, val_pred))

# 4. PCA 차원 축소
scaler = StandardScaler()
pca = PCA(n_components=0.95, random_state=seed)
pipeline = Pipeline([('scaler', scaler), ('pca', pca)])
X_train_pca = pipeline.fit_transform(X_train)
X_val_pca = pipeline.transform(X_val)
print(f"Original: {X_train.shape[1]} features -> PCA: {X_train_pca.shape[1]} components")

# 5. 모델 비교 (원본 vs PCA)
models = {
    'XGB': XGBRegressor(random_state=seed),
    'RF': RandomForestRegressor(random_state=seed),
    'Lasso': Lasso(random_state=seed),
    'SVR': SVR()
}
results = []
for name, model in models.items():
    # 원본
    model.fit(X_train, y_train)
    yv = model.predict(X_val)
    rmse = mean_squared_error(y_val, yv, squared=False)
    r2 = r2_score(y_val, yv)
    # PCA
    model.fit(X_train_pca, y_train)
    yv_pca = model.predict(X_val_pca)
    rmse_pca = mean_squared_error(y_val, yv_pca, squared=False)
    r2_pca = r2_score(y_val, yv_pca)
    results.append({'Model': name,
                    'RMSE_orig': rmse, 'R2_orig': r2,
                    'RMSE_pca': rmse_pca, 'R2_pca': r2_pca})

results_df = pd.DataFrame(results).set_index('Model')
print(results_df)

# 6. 최적 모델(xgb) 저장
best = 'XGB'
best_model = models[best]
best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
joblib.dump(best_model, path + 'best_model.pkl')
print('Best model saved: best_model.pkl')

# 7. 최종 제출 파일 생성
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
