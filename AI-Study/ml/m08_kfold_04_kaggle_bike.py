import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score # fit 과 score가 섞여있음
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor


# 1. 데이터
path = 'C:\\Study25\\_data\\kaggle\\bike\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)           # 가독성을 위해 new_test_csv 로 기입
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

x = train_csv.drop(['count'], axis=1)     
y = train_csv['count']       

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=190) # 분류형태일 때 성능이 더 좋을 때도 있다.
# shuffle=False로 하면 고정이 되지만 데이터가 앞에 50개가 0, 그 다음 1 이런식이라면 라벨 인코딩 이슈가 생길수 있음
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=190) # 분류형태에 사용


# 2. 모델
# model = HistGradientBoostingRegressor()
model = RandomForestRegressor()

# 3. 훈련
print("gannasherry")
scores = cross_val_score(model, x, y, cv=kfold) # fit까지 포함 된거쥬~~?
print('acc : ', scores, '\n평균 acc : ', round(np.mean(scores), 4))

# RandomForestRegressor
#acc :  [0.99973017 0.99983065 0.99974729 0.99946492 0.99978235] 
# 평균 acc :  0.9997

# HistGradientBoostingRegressor
# acc :  [0.99951152 0.99960281 0.99932101 0.9992336  0.99939707] 
# 평균 acc :  0.9994

