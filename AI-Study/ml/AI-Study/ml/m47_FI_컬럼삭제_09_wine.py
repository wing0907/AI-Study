from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
# import xgboost as xgb
# print(xgb.__version__)


seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets['target']

print(x.shape, y.shape)  # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y
)


# 2. 모델구성
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
print("================", model.__class__.__name__, "================")
print('acc : ', model.score(x_test, y_test))
print(model.feature_importances_)

print("25%지점 : ", np.percentile(model.feature_importances_, 25)) # 25% 지점을 출력하게쒈~!
# 0.02461671084165573

percentile =  np.percentile(model.feature_importances_, 25)
print(type(percentile))     # <class 'numpy.float64'>

col_name = []
# 삭제할 컬럼(25% 이하인 놈)을 찾아내자!!!
for i, fi in enumerate(model.feature_importances_): # index 와 값(fi)
    # print(i, fi)
    if fi <= percentile:        # 값이 낮은 놈을 찾아
        col_name.append(datasets.feature_names[i])  # col_name에 집어넣어
    else:
        continue
print(col_name)         # ['ash', 'alcalinity_of_ash', 'nonflavanoid_phenols', 'proanthocyanins']

x_f = pd.DataFrame(x, columns=datasets.feature_names) # 얘는 위에서 만들었어도 된다
x_f = x_f.drop(columns=col_name)                        # 고놈을 삭제해.

print(x)


x_train, x_test = train_test_split(
    x_f, train_size=0.8, random_state=seed,
)

model.fit(x_train, y_train)
print('acc :', model.score(x_test, y_test))         # acc : 0.3055555555555556


# tqdm 간지나는 progress bar

# import matplotlib.pyplot as plt

# def plot_feature_importance_datasets(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     # 수평 막대 그래프, 4개의 열의 feature importance 그래프, 값 위치 센터
#     plt.yticks(np.arange(n_features), model.feature_importances_)
#     # 눈금, 숫자 레이블 표시
#     plt.xlabel("feature Importance")
#     plt.ylabel("Feature")
#     plt.ylim(-1, n_features)        # 축 범위 설정
#     plt.title(model.__class__.__name__)

# plot_feature_importance_datasets(model)
# plt.show()



