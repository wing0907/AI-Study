# pca로 데이터를 축소했을 때 성능이 좋아지는지를 찾는다.
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_wine
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

# 1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 190, stratify=y,
    
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)  # (142, 13) (36, 13)

# 데이터 모아서 한번에 때려보자
# x = np.concatenate([x_train, x_test], axis=0)
# print(x.shape)  # (652, 8)

# exit()
pca = PCA(n_components=13) # 열 최대값
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

# x = x.reshape(70000, 28*28) # (70000, 784)

thresholds = [0.95, 0.99, 0.999, 1.0]

for threshold in thresholds:
    n_components = np.argmax(evr_cumsum >= threshold) + 1
    print(f"{threshold:.3f} 이상을 만족하는 주성분 개수: {n_components}개")

# 0.950 이상을 만족하는 주성분 개수: 10개
# 0.990 이상을 만족하는 주성분 개수: 12개
# 0.999 이상을 만족하는 주성분 개수: 13개
# 1.000 이상을 만족하는 주성분 개수: 13개


# exit()
# 2. PCA 차원 수 리스트
num_components_list = [10, 12, 13]


# 3. DNN 모델 정의 함수
def build_dnn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 4. 반복 실행
for n in num_components_list:
    
    # PCA
    pca = PCA(n_components=n)
    x_pca = pca.fit_transform(x)
    
    # train/test split
    x_train_pca, x_test_pca, y_train_split, y_test_split = train_test_split(
        x_pca, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # DNN 모델 생성 및 학습
    model = build_dnn(n)
    model.fit(x_train_pca, y_train_split, epochs=10, batch_size=128,
              validation_split=0.2, verbose=0)
    
    # 평가
    loss, acc = model.evaluate(x_test_pca, y_test_split, verbose=0)
    print(f"\n▶ n_components = {n}")
    print(f"Test Accuracy: {acc:.4f}")
    
# ▶ n_components = 10
# Test Accuracy: 0.3889

# ▶ n_components = 12
# Test Accuracy: 0.3889

# ▶ n_components = 13
# Test Accuracy: 0.3889