import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 로드
np_path = 'c:/study25/_data/_save_npy/'
x_train = np.load(np_path + "keras44_01_x_train.npy")  # (25000, 100, 100, 3)
y_train = np.load(np_path + "keras44_01_y_train.npy")  # (25000,)
x_test = np.load(np_path + "keras44_01_x_test.npy")    # (12500, 100, 100, 3)
y_test = np.load(np_path + "keras44_01_y_test.npy")    # (12500,)

# 2. reshape
x_train = x_train.reshape(x_train.shape[0], -1)  # (25000, 30000)
x_test = x_test.reshape(x_test.shape[0], -1)      # (12500, 30000)

# 3. Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 4. LDA: 클래스가 100개이면 최대 n_components = 99
n_components = 1
lda = LinearDiscriminantAnalysis(n_components=n_components)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)

# 5. One-hot encoding
y_train_ohe = to_categorical(y_train, num_classes=100)
y_test_ohe = to_categorical(y_test, num_classes=100)

# 6. DNN 모델
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(n_components,)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. 훈련
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
model.fit(x_train_lda, y_train_ohe, validation_split=0.2,
          epochs=100, batch_size=64, callbacks=[es], verbose=1)

# 8. 평가
loss, acc = model.evaluate(x_test_lda, y_test_ohe, verbose=0)
print(f'DNN with LDA({n_components} components) → accuracy: {acc:.4f}')
# DNN with LDA(1 components) → accuracy: 0.4800

