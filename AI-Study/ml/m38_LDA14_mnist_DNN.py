from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# 1. 데이터 로드 및 전처리
(x, y), (_, _) = mnist.load_data()
x = x.reshape(x.shape[0], -1)  # (60000, 784)

# 2. train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337, stratify=y
)

# 3. Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4. LDA: 클래스가 10개이므로 최대 n_components = 9
n_components = 9
lda = LinearDiscriminantAnalysis(n_components=n_components)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)

# 5. One-hot encoding (DNN은 softmax 출력이므로 필요)
y_train_ohe = to_categorical(y_train)
y_test_ohe = to_categorical(y_test)

# 6. DNN 모델 정의
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(n_components,)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))  # MNIST는 10개 클래스

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
model.fit(x_train_lda, y_train_ohe, validation_split=0.2, epochs=100, batch_size=32, callbacks=[es], verbose=1)

# 8. 평가
loss, acc = model.evaluate(x_test_lda, y_test_ohe, verbose=0)
print(f'DNN with LDA({n_components} components) → accuracy: {acc:.4f}')

# DNN with LDA(9 components) → accuracy: 0.9050