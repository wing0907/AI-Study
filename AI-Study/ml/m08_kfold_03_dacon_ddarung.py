import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score # fit 과 score가 섞여있음
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor


# 1. 데이터
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # . = 현재위치, / = 하위폴더
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv.isna().sum())     # 위 함수와 똑같음
train_csv = train_csv.dropna()  #결측치 처리를 삭제하고 남은 값을 반환해 줌
print(test_csv.info())            # test 데이터에 결측치가 있으면 절대 삭제하지 말 것!
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())            # 715 non-null
x = train_csv.drop(['count'], axis=1)    # pandas data framework 에서 행이나 열을 삭제할 수 있다
y = train_csv['count']                   # count 컬럼만 빼서 y 에 넣겠다


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
# acc :  [0.80755007 0.77186078 0.79097921 0.75778814 0.75917929] 
# 평균 acc :  0.7775

# HistGradientBoostingRegressor
# acc :  [0.79918012 0.77709752 0.79439205 0.75457915 0.75158443] 
# 평균 acc :  0.7754

