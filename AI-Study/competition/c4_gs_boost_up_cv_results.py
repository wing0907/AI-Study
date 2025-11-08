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

# 0. 설정
seed = 222
random.seed(seed)
np.random.seed(seed)

# 1. 데이터 로드 및 RDKit 파생변수
path = 'C:/Study25/_data/dacon/boost_up/'
train = pd.read_csv(path + "train.csv")
def rdkit_features(df):
    mols = df['Canonical_Smiles'].apply(Chem.MolFromSmiles)
    df = df.copy()
    df['MolWt'] = mols.apply(Descriptors.MolWt)
    df['TPSA'] = mols.apply(Descriptors.TPSA)
    df['MolLogP'] = mols.apply(Descriptors.MolLogP)
    df['NumRotatableBonds'] = mols.apply(Descriptors.NumRotatableBonds)
    df['NumHAcceptors'] = mols.apply(Descriptors.NumHAcceptors)
    df['NumHDonors'] = mols.apply(Descriptors.NumHDonors)
    return df
train = rdkit_features(train)
feature_cols = ['MolWt','TPSA','MolLogP','NumRotatableBonds','NumHAcceptors','NumHDonors']
X = train[feature_cols]
y = train['Inhibition']

# 2. 훈련/검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

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
print(f"Original features: {X_train.shape[1]} -> PCA components: {X_train_pca.shape[1]}")
print('Explained variance ratio sum:', pca.explained_variance_ratio_.sum())

# 5. 다양한 회귀 알고리즘 비교 (원본 vs PCA)
models = {
    'XGB': XGBRegressor(random_state=seed),
    'RF': RandomForestRegressor(random_state=seed),
    'Lasso': Lasso(random_state=seed),
    'SVR': SVR()
}
results = []
for name, model in models.items():
    # 원본 feature
    model.fit(X_train, y_train)
    yv = model.predict(X_val)
    rmse = mean_squared_error(y_val, yv, squared=False)
    r2 = r2_score(y_val, yv)
    # PCA feature
    model.fit(X_train_pca, y_train)
    yv_pca = model.predict(X_val_pca)
    rmse_pca = mean_squared_error(y_val, yv_pca, squared=False)
    r2_pca = r2_score(y_val, yv_pca)
    results.append({'Model': name,
                    'RMSE_orig': rmse,
                    'R2_orig': r2,
                    'RMSE_pca': rmse_pca,
                    'R2_pca': r2_pca})

results_df = pd.DataFrame(results).set_index('Model')
print(results_df)

# 6. 최적 모델 저장 예시 (XGB 적용)
best = 'XGB'
bst_model = models[best]
bst_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
joblib.dump(bst_model, path + 'best_model.pkl')
print('Best model saved as best_model.pkl')
