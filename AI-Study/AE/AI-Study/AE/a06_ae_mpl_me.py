import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# 재현성(선택)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test  = x_test.reshape(x_test.shape[0], 28*28).astype('float32')/255.

# 노이즈 추가
noise_std = 0.1
x_train_noised = x_train + np.random.normal(0, noise_std, size=x_train.shape)
x_test_noised  = x_test  + np.random.normal(0, noise_std, size=x_test.shape)

# 클리핑(0~1)
x_train_noised = np.clip(x_train_noised, 0, 1)
x_test_noised  = np.clip(x_test_noised, 0, 1)

print("x_train_noised / x_test_noised:", x_train_noised.shape, x_test_noised.shape)
print("x_train max/min:", np.max(x_train), np.min(x_test))
print("x_train_noised max/min:", np.max(x_train_noised), np.min(x_test_noised))

# 2. 세 가지 구조(네가 적어준 그대로)
arch_configs = {
    "Hourglass(모래시계형)": [128, 54, 31, 64, 128],
    "Diamond(다이아몬드형)": [64, 128, 256, 128, 64],
    "Flat(통나무형)":       [128, 128, 128, 128, 128],
}

# 3. 모델 빌더
def build_autoencoder(layer_sizes):
    model = Sequential()
    # 은닉층들(ReLU 권장: 비선형성으로 복원력 향상)
    for i, units in enumerate(layer_sizes):
        if i == 0:
            model.add(Dense(units, input_shape=(784,), activation='relu'))
        else:
            model.add(Dense(units, activation='relu'))
    # 출력층
    model.add(Dense(784, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# 4. 학습 & 예측
EPOCHS = 20
BATCH  = 128
es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)

decoded_by_arch = {}
mse_by_arch = {}

for name, layers in arch_configs.items():
    print(f"\n===== Training {name}: {layers} =====")
    model = build_autoencoder(layers)
    model.fit(
        x_train_noised, x_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[es],
        verbose=1
    )
    decoded = model.predict(x_test_noised, batch_size=256, verbose=0)
    decoded_by_arch[name] = decoded
    mse = np.mean((decoded - x_test) ** 2)
    mse_by_arch[name] = mse
    print(f"[{name}] Test MSE vs clean: {mse:.6f}")

# 5. 시각화: 원본/노이즈/각 구조 복원 비교
n = 5  # 열(샘플) 수
rows = 2 + len(arch_configs)  # 원본/노이즈 + 3개 구조
fig, axes = plt.subplots(rows, n, figsize=(3*n, 2.2*rows))

# 샘플 인덱스 고정
random_images = random.sample(range(x_test.shape[0]), n)

# (Row 1) 원본
for i in range(n):
    ax = axes[0, i] if rows > 1 else axes[i]
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0: ax.set_ylabel('INPUT', size=14)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])

# (Row 2) 노이즈
for i in range(n):
    ax = axes[1, i]
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0: ax.set_ylabel('NOISE', size=14)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])

# (Rows 3~) 각 구조 복원
for r, (name, decoded) in enumerate(decoded_by_arch.items(), start=2):
    for i in range(n):
        ax = axes[r, i]
        ax.imshow(decoded[random_images[i]].reshape(28, 28), cmap='gray')
        if i == 0: ax.set_ylabel(name, size=12)
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.show()

# 6. 성능 요약 출력
print("\n=== 복원 MSE 요약(작을수록 좋음) ===")
for k, v in mse_by_arch.items():
    print(f"{k:>20s}: {v:.6f}")