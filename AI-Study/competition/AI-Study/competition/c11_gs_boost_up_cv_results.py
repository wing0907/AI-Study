import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import joblib
import random
import datetime
import os
import matplotlib.pyplot as plt

# 경고 메시지 제거
RDLogger.DisableLog('rdApp.*')

# 설정
seed = 357
random.seed(seed)
np.random.seed(seed)

# 경로 설정
path = 'C:/Study25/_data/dacon/boost_up/'
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

# SMARTS 기반 기능성 그룹
phenol_group = Chem.MolFromSmarts("c[OH]")
alcohol_group = Chem.MolFromSmarts("[CX4][OH]")
ketone_group = Chem.MolFromSmarts("[CX3](=O)[#6]")
thiol_ether = Chem.MolFromSmarts("[#6]-S-[#6]")
aromatic_thiol_ether = Chem.MolFromSmarts("c-S-[#6]")
sulfonyl_group = Chem.MolFromSmarts("S(=O)(=O)")
phosphates_group = Chem.MolFromSmiles("P(=O)")
amide_group = Chem.MolFromSmarts("[NX3][CX3](=O)[#6]")

# 피처 생성 함수

def rdkit_features(df):
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

    # SMARTS 기반 기능성 그룹
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

# 로그 변환
y_log = np.log1p(y)

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test[feature_cols])

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_log, test_size=0.2, random_state=seed)

# DNN 모델 정의
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 콜백 설정
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = path + f'dnn_model_{now}.h5'
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
]

# 모델 학습
model = build_model(X_scaled.shape[1])
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 모델 저장
model.save(model_path)
print(f"DNN 모델 저장 완료: {model_path}")

# 예측 및 평가
pred_train_log = model.predict(X_train)
pred_val_log = model.predict(X_val)
train_preds = np.expm1(pred_train_log).flatten()
val_preds = np.expm1(pred_val_log).flatten()
y_train_orig = np.expm1(y_train)
y_val_orig = np.expm1(y_val)

train_r2 = r2_score(y_train_orig, train_preds)
val_r2 = r2_score(y_val_orig, val_preds)
train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_preds))
val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_preds))
train_mae = mean_absolute_error(y_train_orig, train_preds)
val_mae = mean_absolute_error(y_val_orig, val_preds)

print(f"Train R² Score: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
print(f"Test  R² Score: {val_r2:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

# 테스트 예측 및 제출 생성
test_pred_log = model.predict(test_scaled)
test_preds = np.expm1(test_pred_log)

submission = pd.DataFrame({
    'ID': test['ID'],
    'Inhibition': test_preds.flatten()
})

csv_path = path + f'submission_dnn_{now}.csv'
submission.to_csv(csv_path, index=False)
print(f"제출 파일 저장 완료: {csv_path}")

# 학습 시각화
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('DNN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()


# Train R² Score: -0.0306, RMSE: 26.9709, MAE: 20.5013
# Test  R² Score: -0.1001, RMSE: 26.9597, MAE: 21.0356
# 4/4 [==============================] - 0s 820us/step
# 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_dnn_20250715_162911.csv


# Train R² Score: -0.0196, RMSE: 26.8453, MAE: 20.5330
# Test  R² Score: -0.1602, RMSE: 27.6003, MAE: 21.2624
# 4/4 [==============================] - 0s 0s/step
# 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_dnn_20250715_165116.csv

