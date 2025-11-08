# PCA Principal Component Analysis 주성분 분석 // #1. 에서만 쓴다
# 컬럼을 줄이는 걸 함. (데이터 전처리) 전처리 개념으로 사용함
# 특성 추출 CNN

# train_test_split 후 scaling 후 pca

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA
import numpy as np
import pandas as pd


# 1. 데이터
path = 'C:\Study25\_data\dacon\따릉이\\'

                # [=] = b 를 a에 넣어줘 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # . = 현재위치, / = 하위폴더
print(train_csv)                  # [1459 rows x 11 columns] -> [1459 rows x 10 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)                   # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission_csv)             # [715 rows x 1 columns]

print(train_csv.shape)            #(1459, 10)
print(test_csv.shape)             #(715, 9)
print(submission_csv.shape)       #(715, 1)

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')           # Nan = 결측치              이상치 ex. 서울의 온도 41도

print(train_csv.info())           # 결측치 확인

print(train_csv.describe())

########################   결측치 처리 1. 삭제   ######################
# print(train_csv.isnull().sum()) # 결측치의 개수 출력
print(train_csv.isna().sum())     # 위 함수와 똑같음

train_csv = train_csv.dropna()  #결측치 처리를 삭제하고 남은 값을 반환해 줌
print(train_csv.isna().sum()) 
print(train_csv.info())         # 결측치 확인
print(train_csv)                # [1328 rows x 10 columns]

########################   결측치 처리 2. 평균값 넣기   ######################
# train_csv = train_csv.fillna(train_csv.mean())
# print(train_csv.isna().sum()) 
# print(train_csv.info()) 

########################   테스트 데이터의 결측치 확인   ######################
print(test_csv.info())            # test 데이터에 결측치가 있으면 절대 삭제하지 말 것!
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())            # 715 non-null

#====================== x 와 y 데이터를 나눠준다 =========================#
x = train_csv.drop(['count'], axis=1)    # pandas data framework 에서 행이나 열을 삭제할 수 있다
                #  count라는 axis=1 열 삭제, 참고로 행 삭제는 axis=0
print(x)                                 # [1459 rows x 9 columns]
y = train_csv['count']                   # count 컬럼만 빼서 y 에 넣겠다
print(y.shape)                           #(1459,)

print(x.shape, y.shape)     # (20640, 8) (20640,)
y_origin = y.copy()

y = np.rint(y).astype(int)
# print(y)
# print(np.unique(y, return_counts=False))

# exit()
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
lda = LinearDiscriminantAnalysis()
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)                  # y는 fit_transform 에서만 넣는다
lda_EVR = lda.explained_variance_ratio_
print(lda_EVR)
print(np.cumsum(lda_EVR))

# [0.22877641 0.13741357 0.12078252 0.09623235 0.09166716 0.08566392
#  0.07271095 0.05976862 0.05843381 0.04855068]

# [0.22877641 0.36618999 0.48697251 0.58320486 0.67487202 0.76053594
#  0.83324689 0.89301551 0.95144932 1.        ]


model = RandomForestRegressor(random_state=333)
model.fit(x_train, y_train_o)
score = model.score(x_test, y_test_o)

print(f'n_components={[lda]} → accuracy: {score:.4f}')

# n_components=[LinearDiscriminantAnalysis()] → accuracy: 0.7060


