import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import math
import matplotlib.pyplot as plt
import joblib
import random

seed = 42
random.seed(seed)
np.random.seed(seed)

# -----------------------------
# 1. 데이터 로드 및 RDKit 파생변수 생성
# -----------------------------
path = 'C:/Study25/_data/dacon/boost_up/'
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

def rdkit_features(df):
    mols = df["Canonical_Smiles"].apply(Chem.MolFromSmiles)
    df = df.copy()
    df["MolWt"] = mols.apply(Descriptors.MolWt)
    df["TPSA"] = mols.apply(Descriptors.TPSA)
    df["MolLogP"] = mols.apply(Descriptors.MolLogP)
    df["NumRotatableBonds"] = mols.apply(Descriptors.NumRotatableBonds)
    df["NumHAcceptors"] = mols.apply(Descriptors.NumHAcceptors)
    df["NumHDonors"] = mols.apply(Descriptors.NumHDonors)
    return df

train = rdkit_features(train)
test = rdkit_features(test)

# -----------------------------
# 2. 훈련/검증 분리
# -----------------------------
feature_cols = ["MolWt", "TPSA", "MolLogP", "NumRotatableBonds", "NumHAcceptors", "NumHDonors"]
X = train[feature_cols]
y = train["Inhibition"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

# -----------------------------
# 3. Pseudo Label 생성 (상위 100개)
# -----------------------------
xgb_temp = XGBRegressor(random_state=seed)
xgb_temp.fit(X_train, y_train)

pseudo_preds = xgb_temp.predict(test[feature_cols])
pseudo_conf = np.abs(pseudo_preds - pseudo_preds.mean())
top_100_idx = np.argsort(-pseudo_conf)[:100]

X_pseudo = test.iloc[top_100_idx][feature_cols]
y_pseudo = pseudo_preds[top_100_idx]

# -----------------------------
# 4. Train + Pseudo 데이터 결합 및 sample_weight 적용
# -----------------------------
X_combined = pd.concat([X, X_pseudo], ignore_index=True)
y_combined = pd.concat([y, pd.Series(y_pseudo)], ignore_index=True)

sample_weights = np.ones(len(X_combined))
sample_weights[-len(X_pseudo):] = 0.3

# -----------------------------
# 5. 기본 파라미터로 최종 모델 훈련
# -----------------------------
final_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=seed
)
final_model.fit(X_combined, y_combined, sample_weight=sample_weights)

# -----------------------------
# 6. 검증 데이터 평가
# -----------------------------
y_val_pred = final_model.predict(X_val)
rmse = math.sqrt(mean_squared_error(y_val, y_val_pred))
r2 = r2_score(y_val, y_val_pred)
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# -----------------------------
# 7. Feature Importance 시각화
# -----------------------------
importances = final_model.feature_importances_
plt.barh(feature_cols, importances)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. 제출 파일 생성
# -----------------------------
test_preds = final_model.predict(test[feature_cols])
submission = pd.DataFrame({
    'ID': test['ID'],
    'Inhibition': test_preds
})
submission.to_csv(path + 'submission_0708_02.csv', index=False)
print("제출 파일 저장 완료: submission_0708_02.csv")

joblib.dump(submission, path + 'c01_0708_02_joblib_save.joblib')

# seed 42
# RMSE: 5.3957
# R2 Score: 0.9581
# 제출 파일 저장 완료: submission_0708_02.csv


# seed 389
# RMSE: 5.4206
# R2 Score: 0.9551
# 제출 파일 저장 완료: submission_0708_01.csv