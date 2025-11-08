from sklearn.datasets import load_diabetes
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
path = 'C:\\Study25\\_data\\kaggle\\bike\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)           # 가독성을 위해 new_test_csv 로 기입
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

x = train_csv.drop(['count'], axis=1)     
y = train_csv['count']    


print(x.shape, y.shape)  

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
model = KNeighborsRegressor(n_neighbors=5)

model.fit(x_train, y_train)
print("================", model.__class__.__name__, "================")
print('loss : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
# acc = accuracy_score(y_test, y_pred)
# print('accuracy_score :', acc)
r2 = r2_score(y_test, y_pred)
print('r2_score :', r2)

# ================ KNeighborsClassifier ================
# loss :  0.9741308500237843
# r2_score : 0.9741308500237843



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



