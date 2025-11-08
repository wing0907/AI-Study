import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing, fetch_covtype
from sklearn.model_selection import KFold, cross_val_score # fit 과 score가 섞여있음
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor


# 1. 데이터
datasets = fetch_covtype(data_home='./fresh_data')

x = datasets.data #(581012, 54) 
y = datasets.target #(581012,)

n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=190) # 분류형태일 때 성능이 더 좋을 때도 있다.
# shuffle=False로 하면 고정이 되지만 데이터가 앞에 50개가 0, 그 다음 1 이런식이라면 라벨 인코딩 이슈가 생길수 있음
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=190) # 분류형태에 사용


# 2. 모델
model = MLPClassifier()
# model = RandomForestClassifier()

# 3. 훈련
print("gannasherry")
scores = cross_val_score(model, x, y, cv=kfold) # fit까지 포함 된거쥬~~?
print('acc : ', scores, '\n평균 acc : ', round(np.mean(scores), 4))

# MLPClassifier
# acc :  [0.75240742 0.765815   0.7638853  0.78050292 0.76803325] 
# 평균 acc :  0.7661

# RandomForestClassifier
# acc :  [0.95617153 0.95516467 0.95391646 0.95546548 0.95606788] 
# 평균 acc :  0.9554

