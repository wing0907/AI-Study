import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping

# 재현성(선택)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1) 데이터 로드 & 전처리 (Conv용: (H, W, C))
(x_train, _), (x_test, _) = mnist.load_data()                    # (60000, 28, 28), (10000, 28, 28)
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=-1)                       # (60000, 28, 28, 1)
x_test  = np.expand_dims(x_test, axis=-1)                        # (10000, 28, 28, 1)

# 노이즈 추가 (가우시안)
noise_std = 0.1
x_train_noised = x_train + np.random.normal(0, noise_std, size=x_train.shape)
x_test_noised  = x_test  + np.random.normal(0, noise_std, size=x_test.shape)

# 클리핑(0~1)
x_train_noised = np.clip(x_train_noised, 0, 1)
x_test_noised  = np.clip(x_test_noised, 0, 1)

print("x_train_noised:", x_train_noised.shape, "x_test_noised:", x_test_noised.shape)

# 2) Conv 오토인코더 정의
# 인코더 (28, 28, 1)
#   Conv (28, 28, 64) padding='same'
#   MaxPool (14, 14, 64)
#   Conv (14, 14, 32) padding='same'
#   MaxPool (7, 7, 32)
# 디코더
#   Conv (7, 7, 32) padding='same'
#   UpSampling2D(2, 2)  -> (14, 14, 32)
#   Conv (14, 14, 16) padding='same'
#   UpSampling2D(2, 2)  -> (28, 28, 16)
#   Conv (28, 28, 1) padding='same'

def build_conv_autoencoder():
    model = Sequential([
        Input(shape=(28, 28, 1)),

        # Encoder
        Conv2D(64, (3, 3), activation='relu', padding='same'),   # -> (28, 28, 64)
        MaxPooling2D((2, 2), padding='same'),                    # -> (14, 14, 64)
        Conv2D(32, (3, 3), activation='relu', padding='same'),   # -> (14, 14, 32)
        MaxPooling2D((2, 2), padding='same'),                    # -> ( 7,  7, 32)

        # Decoder
        Conv2D(32, (3, 3), activation='relu', padding='same'),   # -> ( 7,  7, 32)
        UpSampling2D((2, 2)),                                    # -> (14, 14, 32)
        Conv2D(16, (3, 3), activation='relu', padding='same'),   # -> (14, 14, 16)
        UpSampling2D((2, 2)),                                    # -> (28, 28, 16)
        Conv2D(1, (3, 3), activation='sigmoid', padding='same'), # -> (28, 28,  1)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

model = build_conv_autoencoder()
model.summary()

# 3) 학습
es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
history = model.fit(
    x_train_noised, x_train,
    validation_split=0.2,
    epochs=20,
    batch_size=128,
    callbacks=[es],
    verbose=1
)

# 4) 예측 & 간단한 지표
decoded = model.predict(x_test_noised, batch_size=256, verbose=0)
mse = np.mean((decoded - x_test) ** 2)
print(f"Test MSE vs clean: {mse:.6f}")

# 5) 시각화 (원본/노이즈/복원)
n = 5
fig, axes = plt.subplots(3, n, figsize=(3*n, 7))
idxs = random.sample(range(x_test.shape[0]), n)

# INPUT
for i, idx in enumerate(idxs):
    ax = axes[0, i]
    ax.imshow(x_test[idx].squeeze(), cmap='gray')
    if i == 0: ax.set_ylabel('INPUT', size=14)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

# NOISE
for i, idx in enumerate(idxs):
    ax = axes[1, i]
    ax.imshow(x_test_noised[idx].squeeze(), cmap='gray')
    if i == 0: ax.set_ylabel('NOISE', size=14)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

# OUTPUT
for i, idx in enumerate(idxs):
    ax = axes[2, i]
    ax.imshow(decoded[idx].squeeze(), cmap='gray')
    if i == 0: ax.set_ylabel('OUTPUT', size=14)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

plt.tight_layout()
plt.show()
