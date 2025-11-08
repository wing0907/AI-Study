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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import load_model

# import rdkit as rd
# print(rd.__version__)
# exit()

# 경고 메시지 제거
RDLogger.DisableLog('rdApp.*')

# 0. 설정
seed = 222
random.seed(seed)
np.random.seed(seed)

# 1. 데이터 로드 및 RDKit 파생변수 생성
path = 'C:/Study25/_data/dacon/boost_up/'
train_dir = path+'train.csv'
test_dir = path+'test.csv'
sample_submission_dir = path+'sample_submission.csv'

# 데이터 로드
train_df = pd.read_csv(train_dir,)
test_df = pd.read_csv(test_dir,)
submission_csv = pd.read_csv(sample_submission_dir,)

def get_selected_descriptors(mol):
    selected_descriptors = [
        Descriptors.MolLogP(mol),                                 # LogP
        Descriptors.NumHAcceptors(mol),                           # Number of H-bond acceptors
        Descriptors.NumHDonors(mol),                              # Number of H-bond donors
        Descriptors.TPSA(mol),                                    # Topological Polar Surface Area
        Descriptors.NumAromaticRings(mol),                        # Aromatic rings
        # rdMolDescriptors.CalcFractionCSP3(mol),                   # Fraction of sp3 C
        Lipinski.NumAromaticHeterocycles(mol),                    # Aromatic heterocycles
        Lipinski.NumSaturatedHeterocycles(mol),                   # Saturated heterocycles
        Descriptors.NOCount(mol),                                 # Number of N-O bonds
        # rdMolDescriptors.CalcNumAmideBonds(mol),                  # Number of amide bonds
        Descriptors.NumRadicalElectrons(mol),                     # Number of radical electrons
        Descriptors.NHOHCount(mol),                               # Number of NH or OH groups
    ]
    return selected_descriptors



# ✅ 페놀기 (방향족 고리에 직접 붙은 OH)
phenol_group = Chem.MolFromSmarts("c[OH]")

# ✅ 알코올기 (-OH, 비방향족 탄소에 붙은 OH)
alcohol_group = Chem.MolFromSmarts("[CX4][OH]")

# ✅ 케톤기 (C=O, 양쪽이 탄소일 때)
ketone_group = Chem.MolFromSmarts("[CX3](=O)[#6]")  # 중앙이 C=O, 양쪽이 탄소

# ✅ 티오에터기 (C-S-C)
thiol_ether = Chem.MolFromSmarts("[#6]-S-[#6]")

# ✅ 아릴 티오에터기 (방향족에 결합된 C–S–C)
aromatic_thiol_ether = Chem.MolFromSmarts("c-S-[#6]")  # 방향족 C–S–알킬 C

# ✅ 설포닐기 (-SO2-)
sulfonyl_group = Chem.MolFromSmarts("S(=O)(=O)")
#인산화(P(=O))
phosphates_group = Chem.MolFromSmiles("P(=O)")
# ✅ 아마이드기 (C(=O)N)
amide_group = Chem.MolFromSmarts("[NX3][CX3](=O)[#6]")  # 선택 사항

# Mol 변환
train_df['Mol'] = train_df['Canonical_Smiles'].apply(Chem.MolFromSmiles)
test_df['Mol'] = test_df['Canonical_Smiles'].apply(Chem.MolFromSmiles)

# 기능성 그룹 매칭
train_df['HasPhenol'] = train_df['Mol'].apply(lambda m: m.HasSubstructMatch(phenol_group)).astype(int)
train_df['HasAlcohol'] = train_df['Mol'].apply(lambda m: m.HasSubstructMatch(alcohol_group)).astype(int)
train_df['HasKetone'] = train_df['Mol'].apply(lambda m: m.HasSubstructMatch(ketone_group)).astype(int)
train_df['HasThioether'] = train_df['Mol'].apply(lambda m: m.HasSubstructMatch(thiol_ether)).astype(int)
train_df['HasArylThioether'] = train_df['Mol'].apply(lambda m: m.HasSubstructMatch(aromatic_thiol_ether)).astype(int)
train_df['HasSulfonyl'] = train_df['Mol'].apply(lambda m: m.HasSubstructMatch(sulfonyl_group)).astype(int)
train_df['HasPhosphate'] = train_df['Mol'].apply(lambda m: m.HasSubstructMatch(phosphates_group)).astype(int)
train_df['HasAmide'] = train_df['Mol'].apply(lambda m: m.HasSubstructMatch(amide_group)).astype(int)

test_df['HasPhenol'] = test_df['Mol'].apply(lambda m: m.HasSubstructMatch(phenol_group)).astype(int)
test_df['HasAlcohol'] = test_df['Mol'].apply(lambda m: m.HasSubstructMatch(alcohol_group)).astype(int)
test_df['HasKetone'] = test_df['Mol'].apply(lambda m: m.HasSubstructMatch(ketone_group)).astype(int)
test_df['HasThioether'] = test_df['Mol'].apply(lambda m: m.HasSubstructMatch(thiol_ether)).astype(int)
test_df['HasArylThioether'] = test_df['Mol'].apply(lambda m: m.HasSubstructMatch(aromatic_thiol_ether)).astype(int)
test_df['HasSulfonyl'] = test_df['Mol'].apply(lambda m: m.HasSubstructMatch(sulfonyl_group)).astype(int)
test_df['HasPhosphate'] = test_df['Mol'].apply(lambda m: m.HasSubstructMatch(phosphates_group)).astype(int)
test_df['HasAmide'] = test_df['Mol'].apply(lambda m: m.HasSubstructMatch(amide_group)).astype(int)


descriptor_names = [
    'MolLogP',
    'NumHAcceptors',
    'NumHDonors',
    'TPSA',
    'NumAromaticRings',
    'NumAromaticHeterocycles',
    'NumSaturatedHeterocycles',
    'NOCount',
    'NumRadicalElectrons',
    'NHOHCount'
]

# 각 mol에 대해 get_selected_descriptors() 실행 후 리스트로 반환 -> DataFrame 변환
descriptor_values = train_df['Mol'].apply(get_selected_descriptors).tolist()
descriptor_values_test = test_df['Mol'].apply(get_selected_descriptors).tolist()
