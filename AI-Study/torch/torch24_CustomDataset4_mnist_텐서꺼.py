import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ===========================================
# 1. MNIST 데이터 로드 & 전처리
# ===========================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 정규화 + 차원 추가 (28,28 → 28,28,1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 원-핫 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("학습 데이터:", x_train.shape, y_train.shape)
print("테스트 데이터:", x_test.shape, y_test.shape)

# ===========================================
# 2. 커스텀 데이터셋 클래스
# ===========================================
class MNISTDataset(Sequence):
    def __init__(self, x_data, y_data, batch_size=32, shuffle=True):
        self.x = x_data
        self.y = y_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x))
        self.on_epoch_end()  # 초기 셔플

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))  # 배치 개수

    def __getitem__(self, idx):
        # 배치 인덱스에 맞게 데이터 슬라이싱
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.x[batch_indices], self.y[batch_indices]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ===========================================
# 3. 커스텀 데이터셋 인스턴스화
# ===========================================
train_dataset = MNISTDataset(x_train, y_train, batch_size=64, shuffle=True)
test_dataset = MNISTDataset(x_test, y_test, batch_size=64, shuffle=False)

# ===========================================
# 4. CNN 모델 정의
# ===========================================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ===========================================
# 5. 학습
# ===========================================
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5
)

# ===========================================
# 6. 평가
# ===========================================
test_loss, test_acc = model.evaluate(test_dataset)
print(f"테스트 손실: {test_loss:.4f}, 테스트 정확도: {test_acc:.4f}")

#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 26, 26, 32)        320

#  max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0
#  )

#  conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496

#  max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0
#  2D)

#  flatten (Flatten)           (None, 1600)              0

#  dense (Dense)               (None, 128)               204928

#  dropout (Dropout)           (None, 128)               0

#  dense_1 (Dense)             (None, 10)                1290

# =================================================================
# Total params: 225,034
# Trainable params: 225,034
# Non-trainable params: 0
# _________________________________________________________________
# Epoch 1/5
# 2025-07-30 11:10:36.471556: I tensorflow/stream_executor/cuda/cuda_dnn.cc:366] Loaded cuDNN version 8100
# 2025-07-30 11:10:40.590978: I tensorflow/stream_executor/cuda/cuda_blas.cc:1774] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
# 938/938 [==============================] - 12s 5ms/step - loss: 0.2345 - accuracy: 0.9283 - val_loss: 0.0570 - val_accuracy: 0.9818
# Epoch 2/5
# 938/938 [==============================] - 5s 5ms/step - loss: 0.0843 - accuracy: 0.9750 - val_loss: 0.0377 - val_accuracy: 0.9874
# Epoch 3/5
# 938/938 [==============================] - 5s 5ms/step - loss: 0.0638 - accuracy: 0.9811 - val_loss: 0.0292 - val_accuracy: 0.9895
# Epoch 4/5
# 938/938 [==============================] - 5s 5ms/step - loss: 0.0521 - accuracy: 0.9845 - val_loss: 0.0302 - val_accuracy: 0.9901
# Epoch 5/5
# 938/938 [==============================] - 5s 5ms/step - loss: 0.0425 - accuracy: 0.9870 - val_loss: 0.0237 - val_accuracy: 0.9922
# 157/157 [==============================] - 0s 2ms/step - loss: 0.0237 - accuracy: 0.9922
# 테스트 손실: 0.0237, 테스트 정확도: 0.9922