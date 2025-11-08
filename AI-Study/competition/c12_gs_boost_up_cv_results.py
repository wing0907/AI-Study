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
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import load_model
from keras.optimizers import Adam, SGD, RMSprop
import joblib
import random
import datetime
import os
import matplotlib.pyplot as plt
import keras.backend as K
from sklearn.utils import class_weight
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
test_ids = test['ID'].copy()  # 🔺 ID 백업
test = test[feature_cols]
# 로그 변환
y_log = np.log1p(y)

cols = ['NumHDonors','TPSA','NumRadicalElectrons']

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[cols])
test_scaled = scaler.transform(test[cols])



# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_log, test_size=0.1, random_state=seed)

# RMSE 손실 함수 정의
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# DNN 모델 정의
def build_model(input_dim, optimizer):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=rmse, metrics=['mae'])
    return model

# 옵티마이저 선택 리스트
optimizers = [Adam(learning_rate=0.001), SGD(learning_rate=0.01), RMSprop(learning_rate=0.001)]

# KFold Cross Validation 설정
n_splits = 5  # 폴드 수 (5-Fold Cross Validation)
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

# 콜백 설정
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = path + f'dnn_model_{now}.h5'
callbacks = [
    EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True),
    ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6)  # learning rate 스케줄링 추가
]

# 클래스 가중치 계산
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))




# K-Fold Cross Validation 실행
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_log)):
    print(f"\nTraining fold {fold + 1}/{n_splits}...")

    # 훈련/검증 데이터 분리
    X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
    y_train_fold, y_val_fold = y_log[train_idx], y_log[val_idx]

    # 옵티마이저 선택
    optimizer = optimizers[fold % len(optimizers)]  # 각 폴드마다 다른 옵티마이저 사용

    # 모델 생성
    model = build_model(X_scaled.shape[1], optimizer)

    # 모델 학습
    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=1000,
        batch_size=4,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights
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
    'ID': test_ids,
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

# Train R² Score: -0.1583, RMSE: 28.3781, MAE: 22.3283
# Test  R² Score: -0.0954, RMSE: 27.9028, MAE: 22.0452
# 4/4 [==============================] - 0s 0s/step
# 제출 파일 저장 완료: C:/Study25/_data/dacon/boost_up/submission_dnn_20250716_103629.csv