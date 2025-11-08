# PCA Principal Component Analysis 주성분 분석 // #1. 에서만 쓴다
# 컬럼을 줄이는 걸 함. (데이터 전처리) 전처리 개념으로 사용함
# 특성 추출 CNN

# train_test_split 후 scaling 후 pca

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np


# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337,
    stratify=y
)

### 어떤놈들이 scaler는 pca 하기 전에 하는게 좋댄다.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

for i in range(x.shape[1]):
    pca = PCA(n_components=i+1)   # 몇개의 컬럼을 만들거야
    x_train1 = pca.fit_transform(x_train)
    x_test1 = pca.transform(x_test)    
    # 2. 모델
    model = RandomForestClassifier(random_state=333)

    # 3. 훈련
    model.fit(x_train1, y_train)

    # 4. 평가
    results = model.score(x_test1, y_test)
    print(x_train1.shape, '의 score :', results)

# (120, 1) 의 score : 0.9333333333333333
# (120, 2) 의 score : 0.8333333333333334
# (120, 3) 의 score : 0.9
# (120, 4) 의 score : 0.9666666666666667

evr = pca.explained_variance_ratio_       # 설명가능한 변화율
print("evr : ", evr)    # evr :  [0.73515725 0.22803596 0.0311646  0.00564219]
print("evr_sum : ", sum(evr))   # evr_sum :  0.9999999999999998
# pca 자체에 미치는 영향. 1 돌렸을 때 73%, 2 돌렸을 때 95%
# 약간의 손실은 있다는 뜻.

evr_cumsum = np.cumsum(evr)     # 누적합
print('누적합 : ', evr_cumsum)
# 누적합 :  [0.73515725 0.96319321 0.99435781 1.        ]

# 시각화
import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()

# (120, 1) 의 score : 0.9333333333333333
# (120, 2) 의 score : 0.8333333333333334
# (120, 3) 의 score : 0.9
# (120, 4) 의 score : 0.9666666666666667
# evr :  [0.73515725 0.22803596 0.0311646  0.00564219]
# evr_sum :  0.9999999999999998
# 누적합 :  [0.73515725 0.96319321 0.99435781 1.        ]

exit()




# 2. PCA + 모델 훈련 반복
for n in range(1, 5):  # n_components = 1, 2, 3, 4
    pca = PCA(n_components=n)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    model = RandomForestClassifier(random_state=333)
    model.fit(x_train_pca, y_train)
    score = model.score(x_test_pca, y_test)

    print(f'n_components={n} → accuracy: {score:.4f}')

# n_components=1 → accuracy: 0.9333
# n_components=2 → accuracy: 0.8333
# n_components=3 → accuracy: 0.9000
# n_components=4 → accuracy: 0.9667



