# PCA Principal Component Analysis 주성분 분석 // #1. 에서만 쓴다
# 컬럼을 줄이는 걸 함. (데이터 전처리) 전처리 개념으로 사용함
# 특성 추출 CNN

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target
print(x.shape, y.shape)     # (150, 4) (150,)



### 어떤놈들이 scaler는 pca 하기 전에 하는게 좋댄다.
# scaler = StandardScaler()
# x = scaler.fit_transform(x)


pca = PCA(n_components=1)   # 몇개의 컬럼을 만들거야
x = pca.fit_transform(x)
# print(x)
# print(x.shape)              # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337,
    stratify=y
)

# 2. 모델
model = RandomForestClassifier(random_state=333)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
results = model.score(x_test, y_test)
print(x.shape, '의 score :', results)

# 스케일러 전
# (150, 4) 의 score : 0.9333333333333333
# (150, 3) 의 score : 0.8666666666666667
# (150, 2) 의 score : 0.9
# (150, 1) 의 score : 0.8


# 스케일러 후
# (150, 4) 의 score : 0.9666666666666667
# (150, 3) 의 score : 0.9
# (150, 2) 의 score : 0.8333333333333334
# (150, 1) 의 score : 0.9