import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd

# 1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets['target']

print(x)
print(y)

df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)

n_split = 3
kfold = KFold(n_splits=n_split, shuffle=True)

for idx, (train_index, val_index) in enumerate(kfold.split(df)):  # kfold.split가 2개의 index를 반환해준다
    print("==============[", idx, "], ============")
    print(train_index, '\n', val_index)

# for fold_idx, (train_index, val_index) in enumerate(kfold.split(df), 1):  # 1부터 시작
#     print(f'============== [Fold {fold_idx}] ============')
#     print(train_index, '\n', val_index)