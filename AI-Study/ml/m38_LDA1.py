# PCA Principal Component Analysis 주성분 분석 // #1. 에서만 쓴다
# 컬럼을 줄이는 걸 함. (데이터 전처리) 전처리 개념으로 사용함
# 특성 추출 CNN

# train_test_split 후 scaling 후 pca

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA



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

n_components = 2        # LDA 에서는 y라벨의 개수-1 이하는 가능
n = n_components

lda = LinearDiscriminantAnalysis(n_components=n)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)                  # y는 fit_transform 에서만 넣는다

model = RandomForestClassifier(random_state=333)
model.fit(x_train_lda, y_train)
score = model.score(x_test_lda, y_test)

print(f'n_components={n} → accuracy: {score:.4f}')
# n_components=2 → accuracy: 1.0000


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



