import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()       # x만 준비하고 y는 자리만 명시해주기

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
# DNN 하기 위해 60000, 28*28 로 reshape하고 255.으로 나눈건 scaling 완료
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape) # 평균 0, 표편 0.1인 정규분포 형태의 랜덤값)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape) # 옛날사진 복원하는 작업이라고 생각하면 됨
    
print(x_train_noised.shape, x_test_noised.shape)    # (60000, 784) (10000, 784)
print(np.max(x_train), np.min(x_test))              # 1.0 0.0
print(np.max(x_train_noised), np.min(x_test_noised)) # 1.4941472990656228 -0.5302963833813626 생각보다 넓지만 그리 넓은게 아니다

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, 0, 1)
print(np.max(x_train_noised), np.min(x_test_noised)) # 1.0 0.0


# 2. 모델
input_img = Input(shape=(28*28,))

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(28*28,)))
    model.add(Dense(784, activation='sigmoid'))
    return model

# 실험할 잠재차원 목록 (원래 각각 따로 만들던 것)
hidden_sizes = [1, 8, 32, 64, 154, 331, 486, 713]

models = {}
histories = {}
for h in hidden_sizes:
    print("="*10, f'node {h}개 시작', "="*10)
    m = autoencoder(h)
    m.compile(optimizer='adam', loss='binary_crossentropy')
    hist = m.fit(
        x_train_noised, x_train,
        epochs=20, batch_size=128, validation_split=0.2, verbose=0
    )
    models[h] = m
    histories[h] = hist # 필요하면 나중에 val_loss 비교/그래프 가능

model_01 = autoencoder(hidden_layer_size=1)
model_08 = autoencoder(hidden_layer_size=8)
model_32 = autoencoder(hidden_layer_size=32)
model_64 = autoencoder(hidden_layer_size=64)
model_154 = autoencoder(hidden_layer_size=154)
model_331 = autoencoder(hidden_layer_size=331)
model_486 = autoencoder(hidden_layer_size=486)
model_713 = autoencoder(hidden_layer_size=713)

# 실험할 잠재차원 목록 (원래 각각 따로 만들던 것)
hidden_sizes = [1, 8, 32, 64, 154, 331, 486, 713]

# 3. 학습: for문으로 모델 생성/컴파일/훈련을 한 번에
models = {}
histories = {}
for h in hidden_sizes:
    print("="*10, f'node {h}개 시작', "="*10)
    m = autoencoder(h)
    m.compile(optimizer='adam', loss='binary_crossentropy')
    hist = m.fit(
        x_train_noised, x_train,
        epochs=20, batch_size=128, validation_split=0.2, verbose=0
    )
    models[h] = m
    histories[h] = hist  # 필요하면 나중에 val_loss 비교/그래프 가능


# 4. 예측: for문으로 모든 모델 예측 저장
decoded_by_h = {}
for h, m in models.items():
    decoded_by_h[h] = m.predict(x_test_noised, verbose=0)

# 5. 시각화: 원본 + 각 잠재차원 결과를 for문으로 그리기
num_cols = 5
num_rows = 1 + len(hidden_sizes)  # 첫 줄: 원본, 이후: 각 모델
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 2 * num_rows))

# 한 번 뽑은 5개 인덱스를 모든 행에서 재사용
random.seed(0)
random_indices = random.sample(range(x_test.shape[0]), num_cols)

# 첫 번째 행: 원본(노이즈/클린 중 원하는 걸 보여줘도 됨)
for col_idx, idx in enumerate(random_indices):
    ax = axes[0, col_idx]
    ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    ax.set_xticks([]); ax.set_yticks([])
    if col_idx == 0:
        ax.set_ylabel("Original", rotation=0, labelpad=35, va='center')

# 다음 행들: 각 잠재차원 모델의 복원 이미지
for row_idx, h in enumerate(hidden_sizes, start=1):
    decoded = decoded_by_h[h]
    for col_idx, idx in enumerate(random_indices):
        ax = axes[row_idx, col_idx]
        ax.imshow(decoded[idx].reshape(28, 28), cmap='gray')
        ax.set_xticks([]); ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel(f"k={h}", rotation=0, labelpad=35, va='center')

plt.tight_layout()
plt.show()


