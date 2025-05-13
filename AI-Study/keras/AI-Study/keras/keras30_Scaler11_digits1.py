# 필요한 라이브러리 설치 (Colab 또는 로컬에서 최초 한 번만)
# pip install tensorflow scikit-learn matplotlib seaborn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 로드 및 전처리
digits = load_digits()
X = digits.images.reshape(-1, 8, 8, 1).astype("float32") / 16.0
y = to_categorical(digits.target, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=digits.target, random_state=47
)

# 2. CNN 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. 모델 학습
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[es])

# 4. 모델 평가
y_test_prob = model.predict(X_test)
y_test_label = np.argmax(y_test, axis=1)
y_pred_label = np.argmax(y_test_prob, axis=1)
logloss = log_loss(y_test, y_test_prob)
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

print(f"✅ Test Log Loss: {logloss:.4f}")
print(f"✅ Test Accuracy: {accuracy:.4f}")
print(f"✅ Total Parameters: {model.count_params()}")

# 5. 예측 샘플 시각화
plt.figure(figsize=(10, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"True: {y_test_label[i]}\nPred: {y_pred_label[i]}")
    plt.axis('off')
plt.suptitle("🔍 Sample Predictions")
plt.tight_layout()
plt.show()

# 6. Confusion Matrix 시각화
cm = confusion_matrix(y_test_label, y_pred_label)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("📊 Confusion Matrix")
plt.show()
