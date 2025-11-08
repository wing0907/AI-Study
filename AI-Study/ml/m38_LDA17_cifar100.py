# PCA Principal Component Analysis 주성분 분석 // #1. 에서만 쓴다
# 컬럼을 줄이는 걸 함. (데이터 전처리) 전처리 개념으로 사용함
# 특성 추출 CNN

# train_test_split 후 scaling 후 pca

from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA
import numpy as np


# 1. 데이터 로드 및 전처리
(x, y), (_, _) = cifar100.load_data()            # 여기서 test는 사용 안 함
x = x.reshape(x.shape[0], -1)                 # (60000, 784)



# (_, y), _ = cifar100.load_data()
# print("라벨의 종류:", np.unique(y))
# print("클래스 개수:", len(np.unique(y)))

# 라벨의 종류: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
#  48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
#  72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
#  96 97 98 99]
# 클래스 개수: 100

# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337,
    stratify=y
)

### 어떤놈들이 scaler는 pca 하기 전에 하는게 좋댄다.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_components = 99        # LDA 에서는 y라벨의 개수-1 이하는 가능
n = n_components

lda = LinearDiscriminantAnalysis(n_components=n)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)                  # y는 fit_transform 에서만 넣는다

model = RandomForestClassifier(random_state=333)
model.fit(x_train_lda, y_train)
score = model.score(x_test_lda, y_test)

print(f'n_components={n} → accuracy: {score:.4f}')
# n_components=99 → accuracy: 0.1251


exit()
# 2. PCA + 모델 훈련 반복
for n in range(1, 5):  # n_components = 1, 2, 3, 4
    lda = LinearDiscriminantAnalysis(n_components=n)
    x_train_lda = lda.fit_transform(x_train, y_train)
    x_test_lda = lda.transform(x_test, y_test)

    model = RandomForestClassifier(random_state=333)
    model.fit(x_train_lda, y_train)
    score = model.score(x_test_lda, y_test)

    print(f'n_components={n} → accuracy: {score:.4f}')

# n_components=1 → accuracy: 0.9333
# n_components=2 → accuracy: 0.8333
# n_components=3 → accuracy: 0.9000
# n_components=4 → accuracy: 0.9667



