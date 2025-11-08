from sklearn.datasets import load_breast_cancer
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)  

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
# model = KNeighborsClassifier(n_neighbors=5)
model = KMeans(n_clusters=2, init='k-means++', 
               n_init=10, random_state=seed)

y_train_pred = model.fit_predict(x_train)

print(y_train_pred[:10])
print(y_train[:10])

from sklearn.metrics import confusion_matrix

# 클러스터 예측
y_train_pred = model.fit_predict(x_train)

# 혼동 행렬 출력
cm = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix:\n", cm)


# 클러스터 예측
y_train_pred = model.fit_predict(x_train)

# confusion matrix
cm = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix:\n", cm)

# 정확하게 대응되는 방향 선택
acc1 = accuracy_score(y_train, y_train_pred)
acc2 = accuracy_score(y_train, 1 - y_train_pred)

if acc1 > acc2:
    print("클러스터 라벨은 실제 y와 동일하게 매핑됨")
    y_aligned = y_train_pred
else:
    print("클러스터 라벨은 실제 y와 반대로 매핑됨 (0↔1)")
    y_aligned = 1 - y_train_pred

# 정렬된 예측값으로 최종 정확도
print("최종 Accuracy:", accuracy_score(y_train, y_aligned))


import pandas as pd

df = pd.DataFrame({'Actual': y_train, 'Cluster': y_train_pred})
print(pd.crosstab(df['Actual'], df['Cluster'], rownames=['Actual'], colnames=['Predicted Cluster']))


# Confusion Matrix:
#  [[138  32]
#  [ 11 274]]

# 클러스터 라벨은 실제 y와 동일하게 매핑됨
# 최종 Accuracy: 0.9054945054945055

# Predicted Cluster    0    1
# Actual
# 0                  138   32
# 1                   11  274



scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)

# 2. KMeans 클러스터링
model = KMeans(n_clusters=2, random_state=seed)
y_pred = model.fit_predict(x_scaled)

# 3. PCA로 차원 축소
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_train)


# 4. 시각화
plt.figure(figsize=(12, 5))

# 실제 라벨 시각화
plt.subplot(1, 2, 1)
plt.title("Actual Labels")
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y_train, cmap='coolwarm', s=15)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)

# 예측된 클러스터 시각화
plt.subplot(1, 2, 2)
plt.title("KMeans Predicted Clusters")
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y_pred, cmap='coolwarm', s=15)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)

plt.tight_layout()
plt.show()


exit()


model.fit(x_train, y_train)
print("================", model.__class__.__name__, "================")
print('acc : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy_score :', acc)
f1 = f1_score(y_test, y_pred)
print('f1_score :', f1)

# ================ KNeighborsClassifier ================
# acc :  0.9736842105263158
# accuracy_score : 0.9736842105263158
# f1_score : 0.9793103448275862



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



