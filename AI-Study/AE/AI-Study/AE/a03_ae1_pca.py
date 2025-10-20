# import numpy as np
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Input
# from sklearn.decomposition import PCA

# # 1. 데이터
# (x_train, _), (x_test, _) = mnist.load_data()       # x만 준비하고 y는 자리만 명시해주기

# x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
# # DNN 하기 위해 60000, 28*28 로 reshape하고 255.으로 나눈건 scaling 완료
# x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.

# x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape) # 평균 0, 표편 0.1인 정규분포 형태의 랜덤값)
# x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape) # 옛날사진 복원하는 작업이라고 생각하면 됨
    
# print(x_train_noised.shape, x_test_noised.shape)    # (60000, 784) (10000, 784)
# print(np.max(x_train), np.min(x_test))              # 1.0 0.0
# print(np.max(x_train_noised), np.min(x_test_noised)) # 1.4941472990656228 -0.5302963833813626 생각보다 넓지만 그리 넓은게 아니다

# x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
# x_test_noised = np.clip(x_test_noised, 0, 1)
# print(np.max(x_train_noised), np.min(x_test_noised)) # 1.0 0.0




# # 2. 모델
# input_img = Input(shape=(28*28,))

# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Dense(units=hidden_layer_size, input_shape=(28*28,)))
#     model.add(Dense(784, activation='sigmoid'))
#     return model

# # 0.95 이상 :
# # 0.99 이상 :
# # 0.999 이상 :
# # 1.0 일때 :



# hidden_size = 64

# model = autoencoder(hidden_layer_size=hidden_size)

# # 3. 컴파일, 훈련
# # model.compile(optimizer='adam', loss='mse')
# model.compile(optimizer='adam', loss='binary_crossentropy') # mse 보다 잘 보임. 결과치 보고 성능 판단ㄱㄱ
# model.fit(x_train_noised, x_train, epochs=50, batch_size=128, validation_split=0.2)

# # 4. 평가, 예측
# decoded_img = model.predict(x_test_noised)

# import matplotlib.pyplot as plt
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     ax = plt.subplot(2, n, i+1)
#     plt.imshow(x_test_noised[i].reshape(28,28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     ax = plt.subplot(2, n, i+1+n)
#     plt.imshow(decoded_img[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()



# ============================
# MNIST Denoising Autoencoder + PCA 차원선택 전체 코드
# ============================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# 0. 재현성 고정(선택)
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# 1. 데이터 로드 & 전처리
# ----------------------------
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32') / 255.0
x_test  = x_test.reshape(x_test.shape[0], 28*28).astype('float32') / 255.0

# 노이즈 추가(가우시안)
noise_std = 0.1
x_train_noised = x_train + np.random.normal(0, noise_std, size=x_train.shape)
x_test_noised  = x_test  + np.random.normal(0, noise_std, size=x_test.shape)

# 클리핑(0~1)
x_train_noised = np.clip(x_train_noised, 0, 1)
x_test_noised  = np.clip(x_test_noised, 0, 1)

print("x_train_noised / x_test_noised:", x_train_noised.shape, x_test_noised.shape)
print("x_train max/min:", np.max(x_train), np.min(x_test))
print("x_train_noised max/min:", np.max(x_train_noised), np.min(x_test_noised))

# ----------------------------
# 2. PCA로 임계값별 필요 차원 수 계산
#    (오토인코더 숨은층 크기 가이드)
# ----------------------------
X = x_train  # 보통 '클린 데이터'로 PCA 기준을 잡습니다.

targets = [0.95, 0.99, 0.999]  # 1.0은 전체 차원(784)로 처리
need = {}

# 방법 1: n_components=비율 로 직접 피팅
for t in targets:
    pca = PCA(n_components=t, svd_solver='full', whiten=False, random_state=SEED)
    pca.fit(X)
    need[t] = pca.n_components_


# 방법 2: 전체 성분 적합 후 누적합에서 직접 찾기(검증용)
pca_full = PCA(n_components=None, svd_solver='full', whiten=False, random_state=SEED)
pca_full.fit(X)
cum = np.cumsum(pca_full.explained_variance_ratio_)
need_alt = {t: int(np.searchsorted(cum, t) + 1) for t in targets}

# 1.0(100%)는 일반적으로 전체 차원(=784)
full_dim = X.shape[1]

print("\n[PCA 필요 차원 수]")
print(f"≥0.95 : {need[0.95]} (alt: {need_alt[0.95]})")
print(f"≥0.99 : {need[0.99]} (alt: {need_alt[0.99]})")
print(f"≥0.999: {need[0.999]} (alt: {need_alt[0.999]})")
print(f"=1.0  : {full_dim}")

# ----------------------------
# 3. 오토인코더 정의 함수
#    (선형 bottleneck + sigmoid 출력)
# ----------------------------
def build_autoencoder(hidden_units: int) -> Sequential:
    model = Sequential([
        Dense(hidden_units, input_shape=(784,), activation=None),  # 선형 bottleneck (PCA 유사)
        Dense(784, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# ----------------------------
# 4. 임계값별로 모델 학습
#    (ES로 과적합/시간 방지)
# ----------------------------
settings = [
    ("0.95",  need[0.95]),
    ("0.99",  need[0.99]),
    ("0.999", need[0.999]),
    ("1.0",   full_dim),
]

histories = {}
models = {}
recons = {}

es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
EPOCHS = 20
BATCH  = 128

for label, hidden_size in settings:
    print(f"\n=== Training AE for {label} (hidden={hidden_size}) ===")
    model = build_autoencoder(hidden_size)
    hist = model.fit(
        x_train_noised, x_train,
        validation_split=0.2,
        epochs=EPOCHS, batch_size=BATCH,
        callbacks=[es], verbose=1
    )
    models[label] = model
    histories[label] = hist

    # 복원
    decoded = model.predict(x_test_noised, batch_size=256, verbose=0)
    recons[label] = decoded

    # 간단한 평가지표 (MSE)
    mse = np.mean((decoded - x_test) ** 2)
    print(f"[{label}] Test MSE vs clean: {mse:.6f}")

# ----------------------------
# 5. 결과 시각화
#    첫 행: 노이즈 입력
#    이후 행: 임계값별 복원 결과
# ----------------------------
n = 10  # 보여줄 샘플 수
rows = 1 + len(settings)
plt.figure(figsize=(2*n, 2*rows))

# (Row 1) 노이즈 입력
for i in range(n):
    ax = plt.subplot(rows, n, i + 1)
    plt.imshow(x_test_noised[i].reshape(28, 28))
    plt.gray()
    ax.set_title("Noisy", fontsize=10)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# (Rows 2~) 복원 결과
for r, (label, hidden_size) in enumerate(settings, start=2):
    decoded_img = recons[label]
    for i in range(n):
        ax = plt.subplot(rows, n, (r - 1) * n + i + 1)
        plt.imshow(decoded_img[i].reshape(28, 28))
        plt.gray()
        if i == 0:
            ax.set_title(f"AE {label}\n(hidden={hidden_size})", fontsize=10)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.suptitle("MNIST Denoising: PCA-guided Hidden Sizes", fontsize=14)
plt.tight_layout()
plt.show()

# ----------------------------
# 6. 콘솔에 최종 요약 출력
# ----------------------------
print("\n=== 최종 요약 ===")
print(f"0.95 이상 : {need[0.95]}")
print(f"0.99 이상 : {need[0.99]}")
print(f"0.999 이상: {need[0.999]}")
print(f"1.0 일때 : {full_dim} (전체 차원)")

# === 최종 요약 ===
# 0.95 이상 : 154
# 0.99 이상 : 331
# 0.999 이상: 486
# 1.0 일때 : 784 (전체 차원)