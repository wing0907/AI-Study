from sklearn.datasets import fetch_covtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
# import xgboost as xgb
# print(xgb.__version__)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)



x = train_csv.drop(['target'], axis=1)
y = train_csv['target']


# 2. 라벨 인코딩
le = LabelEncoder()
y_enc = le.fit_transform(y)   # now 0~8

# 3. 초기 train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y_enc, train_size=0.8, random_state=seed, stratify=y_enc
)

# 4. 모델 학습 (전체 feature)
model = XGBClassifier(random_state=seed)
model.fit(x_train, y_train)
print("전체 사용 ACC:", model.score(x_test, y_test))

# 5. Feature Importance 기반 하위 25% 컬럼 인덱스 추출
importances = model.feature_importances_
th = np.percentile(importances, 25)
low_idx = [i for i, imp in enumerate(importances) if imp <= th]
print("제거할 컬럼 인덱스:", low_idx)

# 6. 컬럼 제거
x_reduced = x.drop(columns=[x.columns[i] for i in low_idx])

# 7. 제거된 특성으로 다시 split (y도 함께!)
x_tr_r, x_te_r, y_tr_r, y_te_r = train_test_split(
    x_reduced, y_enc, train_size=0.8, random_state=seed, stratify=y_enc
)

# 8. 재학습 & 평가
model.fit(x_tr_r, y_tr_r)
print("제거 후 ACC:", model.score(x_te_r, y_te_r))


# 전체 사용 ACC: 0.8116515837104072
# 제거할 컬럼 인덱스: [4, 5, 9, 11, 17, 20, 21, 27, 30, 43, 45, 48, 53, 60, 62, 64, 65, 69, 72, 73, 80, 81, 86, 88]
# 제거 후 ACC: 0.8106819650937298



# # 2. 모델구성
# model = XGBClassifier(random_state=seed)

# model.fit(x_train, y_train)
# print("================", model.__class__.__name__, "================")
# print('acc : ', model.score(x_test, y_test))
# print(model.feature_importances_)

# print("25%지점 : ", np.percentile(model.feature_importances_, 25)) # 25% 지점을 출력하게쒈~!
# # 25%지점 :  0.006563458708114922

# percentile =  np.percentile(model.feature_importances_, 25)
# print(type(percentile))     # <class 'numpy.float64'>

# col_name = []
# # 삭제할 컬럼(25% 이하인 놈)을 찾아내자!!!
# for i, fi in enumerate(model.feature_importances_): # index 와 값(fi)
#     # print(i, fi)
#     if fi <= percentile:        # 값이 낮은 놈을 찾아
#         col_name.append(i)  # col_name에 집어넣어
#     else:
#         continue
# print(col_name)         # [2, 8, 14, 18, 19, 20, 27, 28, 31, 32, 39, 41, 47, 49]

# x_f = pd.DataFrame(x) # 얘는 위에서 만들었어도 된다
# x_f = x_f.drop(columns=col_name)                        # 고놈을 삭제해.

# print(x_f)


# x_train, x_test = train_test_split(
#     x_f, train_size=0.8, random_state=seed,
# )

# model.fit(x_train, y_train)                         # acc :  0.8686264554271405
# print('acc :', model.score(x_test, y_test))         # acc : 0.48491002814040945


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



