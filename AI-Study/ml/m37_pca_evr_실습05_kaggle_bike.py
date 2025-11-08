# pca로 데이터를 축소했을 때 성능이 좋아지는지를 찾는다.
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

# 1. 데이터
path = 'C:\\Study25\\_data\\kaggle\\bike\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)           # 가독성을 위해 new_test_csv 로 기입
submission_csv = pd.read_csv(path + 'sampleSubmission.csv',)

print(train_csv)                # [10886 rows x 11 columns]
print(new_test_csv)                 # [6493 rows x 10 columns]
print(train_csv.shape)          # (10886, 11)
print(new_test_csv.shape)           # (6493, 10)
print(submission_csv.shape)     # (6493, 2)

x = train_csv.drop(['count'], axis=1)       # [10886 rows x 10 columns]
y = train_csv['count']                      # (10886,)
print(x)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 190,
    
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)  # (8708, 10) (2178, 10)

# 데이터 모아서 한번에 때려보자
x = np.concatenate([x_train, x_test], axis=0)
print(x.shape)  # (10886, 10)

# exit()
pca = PCA(n_components=10) # 열 최대값
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

# x = x.reshape(70000, 28*28) # (70000, 784)

thresholds = [0.95, 0.99, 0.999, 1.0]

for threshold in thresholds:
    n_components = np.argmax(evr_cumsum >= threshold) + 1
    print(f"{threshold:.3f} 이상을 만족하는 주성분 개수: {n_components}개")

# 0.950 이상을 만족하는 주성분 개수: 7개
# 0.990 이상을 만족하는 주성분 개수: 9개
# 0.999 이상을 만족하는 주성분 개수: 9개
# 1.000 이상을 만족하는 주성분 개수: 10개


# exit()
# 2. PCA 차원 수 리스트
num_components_list = [7, 9, 10]


# 3. DNN 모델 정의 함수
def build_dnn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam',)
    return model

# 4. 반복 실행
for n in num_components_list:
    print(f"\n▶ n_components = {n}")
    
    # PCA
    pca = PCA(n_components=n)
    x_pca = pca.fit_transform(x)
    
    # train/test split
    x_train_pca, x_test_pca, y_train_split, y_test_split = train_test_split(
        x_pca, y, test_size=0.2, random_state=42, #stratify=np.argmax(y, axis=1)
    )
    
    # DNN 모델 생성 및 학습
    model = build_dnn(n)
    model.fit(x_train_pca, y_train_split, epochs=10, batch_size=128,
              validation_split=0.2, verbose=0)
    
    # 평가
    loss = model.evaluate(x_test_pca, y_test_split, verbose=0)
    # 예측
    y_pred = model.predict(x_test_pca, verbose=0)
    
    # 성능 평가
    r2 = r2_score(y_test_split, y_pred)
    print(f"R2 Score: {r2:.4f}")
    
# ▶ n_components = 7
# R2 Score: -0.0028

# ▶ n_components = 9
# R2 Score: -0.0029

# ▶ n_components = 10
# R2 Score: -0.0031