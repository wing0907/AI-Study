import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터 로드
np_path = 'c:/study25/_data/_save_npy/'
x_train = np.load(np_path + "keras44_01_x_train.npy")  # (25000, 100, 100, 3)
y_train = np.load(np_path + "keras44_01_y_train.npy")  # (25000,)
x_test = np.load(np_path + "keras44_01_x_test.npy")    # (12500, 100, 100, 3)
y_test = np.load(np_path + "keras44_01_y_test.npy")    # (12500,)

# 2. reshape
x_train = x_train.reshape(x_train.shape[0], -1)  # (25000, 30000)
x_test = x_test.reshape(x_test.shape[0], -1)      # (12500, 30000)

# print("라벨의 종류:", np.unique(y))
# print("클래스 개수:", len(np.unique(y)))

# 3. scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4. LDA
n_components = 1  # binary classification (cat vs dog)
lda = LinearDiscriminantAnalysis(n_components=n_components)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)

# 5. 모델 훈련 및 평가
model = RandomForestClassifier(random_state=333)
model.fit(x_train_lda, y_train)
score = model.score(x_test_lda, y_test)

print(f'LDA({n_components} components) → accuracy on test set: {score:.4f}')
# LDA(1 components) → accuracy: 0.5485