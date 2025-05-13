# PCA Principal Component Analysis 주성분 분석 // #1. 에서만 쓴다
# 컬럼을 줄이는 걸 함. (데이터 전처리) 전처리 개념으로 사용함
# 특성 추출 CNN

# train_test_split 후 scaling 후 pca

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA
import numpy as np



# 1. 데이터
datasets = load_diabetes()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)     # (442, 10) (442,)
y_origin = y.copy()

y = np.rint(y).astype(int)
# print(y)
# print(np.unique(y, return_counts=False))


x_train, x_test, y_train, y_test, y_train_o, y_test_o = train_test_split(
    x, y, y_origin, test_size=0.2, random_state=337,
    # stratify=y
)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


###################### PCA ##########################
# pca = PCA(n_components=10)
# x_train = pca.fit_transform(x_train)
# x_test = pca.transform(x_test)
# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(np.cumsum(pca_EVR))

# [0.41116448 0.14917514 0.11659554 0.09821976 0.06493619 0.05734679
#  0.05236123 0.04194557 0.00749355 0.00076175]

# [0.41116448 0.56033961 0.67693515 0.77515491 0.8400911  0.89743789
#  0.94979913 0.9917447  0.99923825 1.        ]

###################### LDA ##########################
# lda = LinearDiscriminantAnalysis(n_components=10)
# x_train = lda.fit_transform(x_train, y_train)
# x_test = lda.transform(x_test)                  # y는 fit_transform 에서만 넣는다
# lda_EVR = lda.explained_variance_ratio_
# print(lda_EVR)
# print(np.cumsum(lda_EVR))

# [0.22877641 0.13741357 0.12078252 0.09623235 0.09166716 0.08566392
#  0.07271095 0.05976862 0.05843381 0.04855068]

# [0.22877641 0.36618999 0.48697251 0.58320486 0.67487202 0.76053594
#  0.83324689 0.89301551 0.95144932 1.        ]


model = RandomForestRegressor(random_state=333)
model.fit(x_train, y_train_o)
score = model.score(x_test, y_test_o)

print(f'n_components={[]} → accuracy: {score:.4f}')

# n_components=[] → accuracy: 0.4141
# n_components=PCA(n_components=10) → accuracy: 0.3747
# n_components=LinearDiscriminantAnalysis(n_components=10) → accuracy: 0.4597

