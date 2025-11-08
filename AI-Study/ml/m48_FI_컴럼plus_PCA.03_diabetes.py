from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
# import xgboost as xgb
# print(xgb.__version__)


seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y
)


# 2. 모델구성
model = XGBRegressor(random_state=seed)

model.fit(x_train, y_train)
print("================", model.__class__.__name__, "================")
print('r2 : ', model.score(x_test, y_test))
print(model.feature_importances_)
# acc :  0.83707103301617

# [0.47826383 0.07366086 0.0509511  0.02446287 0.02366972 0.14824368
#  0.0921493  0.10859864]
print("25%지점 : ", np.percentile(model.feature_importances_, 25)) # 25% 지점을 출력하게쒈~!
# 25%지점 :  0.044329043477773666


percentile =  np.percentile(model.feature_importances_, 25)
# print(type(percentile))     # <class 'numpy.float64'>

col_name = []
# 삭제할 컬럼(25% 이하인 놈)을 찾아내자!!!
for i, fi in enumerate(model.feature_importances_): # index 와 값(fi)
    # print(i, fi)
    if fi <= percentile:        # 값이 낮은 놈을 찾아
        col_name.append(datasets.feature_names[i])  # col_name에 집어넣어
    else:
        continue
print(col_name)         # ['age', 's1', 's3']   얘네를 PCA로 압축해서 붙여버리겠드아!

# exit()

x_f = pd.DataFrame(x, columns=datasets.feature_names) # 얘는 위에서 만들었어도 된다
x1 = x_f.drop(columns=col_name)                        # 고놈을 삭제해.
x2 = x_f[['age', 's1', 's3']]
print(x2)

x1_train, x1_test, x2_train, x2_test,  = train_test_split(
    x1, x2, train_size=0.8, random_state=seed,
    # stratify=y
)
print(x1_train.shape, x1_test.shape)    # (16512, 6) (4128, 6)
print(x2_train.shape, x2_test.shape)    # (16512, 2) (4128, 2)
print(y_train.shape, y_test.shape)      # (16512,) (4128,)

pca = PCA(n_components=1)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)
print(x2_train.shape, x2_test.shape)    # (16512, 1) (4128, 1)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)
print(x_train.shape, x_test.shape)      # (16512, 7) (4128, 7)


model.fit(x_train, y_train)                                   # r2 :  0.39065385219018145
print('FI_Drop + PCA :', model.score(x_test, y_test))         # FI_Drop : 0.5164226445137823
                                                              # FI_Drop + PCA : 0.4160264085258457

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



