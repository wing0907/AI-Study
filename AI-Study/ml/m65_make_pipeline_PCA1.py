import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


# 1. 데이터
x, y = load_digits(return_X_y=True)
print(x.shape, y.shape) # (1797, 64) (1797,)

pca = PCA(n_components=8)
x = pca.fit_transform(x)
print(x.shape)  # (1797, 8) 차원 축소


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=333,
    stratify=y
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# # 2. 모델
# model = RandomForestClassifier()
# model = make_pipeline(StandardScaler(), RandomForestClassifier())
# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model = make_pipeline(PCA(n_components=8), MinMaxScaler(), SVC())
# 스케일러 다음 모델이 나오는 순서를 지켜줘야 한다.


# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('model.score:', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score:', acc)



