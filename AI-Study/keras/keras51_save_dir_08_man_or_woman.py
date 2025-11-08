from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
matplotlib.rcParams['axes.unicode_minus'] = False 

# ---------------------------
# 1. 데이터 로드 및 분리
# ---------------------------
np_path = 'c:/study25/_data/_save_npy/'

start = time.time()
x1_train = np.load(np_path + "keras46_07_x_train.npy")
y1_train = np.load(np_path + "keras46_07_y_train.npy")
end = time.time()

x_train, x_test, y_train, y_test = train_test_split(
    x1_train, y1_train, test_size=0.3, random_state=333, shuffle=True)

# ---------------------------
# 2. 데이터 형태 변형 및 정규화
# ---------------------------
x_train = x_train.reshape(-1, 250, 250, 3)
x_test = x_test.reshape(-1, 250, 250, 3)

x_train = x_train / 255.
x_test = x_test / 255.

# ---------------------------
# 3. 클래스 0만 증강하기
# ---------------------------
zero_idx = np.where(y_train == 0)[0]
x_zero = x_train[zero_idx]
y_zero = y_train[zero_idx]

augment_size = 8000  # 원하는 증강 수
randidx = np.random.randint(len(x_zero), size=augment_size)
x_aug = x_zero[randidx].copy()
y_aug = y_zero[randidx].copy()

datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    rotation_range=15,
    fill_mode='nearest'
)

x_augmented = datagen.flow(
    x_aug,
    y_aug,
    batch_size=augment_size,
    shuffle=False,
    save_prefix='man_or_woman',
    save_to_dir='C:\Study25\_data\_save_img\\08_man_or_woman\\',
).next()[0]

exit()

# ---------------------------
# 4. 증강된 데이터 합치기
# ---------------------------
x_train = np.concatenate([x_train, x_augmented])
y_train = np.concatenate([y_train, y_aug])

print("최종 x_train:", x_train.shape)
print("최종 y_train 분포:", np.unique(y_train, return_counts=True))

# ---------------------------
# 5. 모델 로드
# ---------------------------
path = './_save/horse_or_human/'
model = load_model(path + 'k46_08_1.hdf5')

# ---------------------------
# 6. 모델 컴파일 및 학습
# ---------------------------
from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30, verbose=1,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=500, batch_size=512,
          validation_split=0.2,
          callbacks=[es],
          verbose=1)

# ---------------------------
# 7. 평가
# ---------------------------
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])

# ---------------------------
# 8. 예측 및 정확도
# ---------------------------
y_pred = model.predict(x_test)
y_pred_binary = (y_pred > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred_binary)
print('acc:', round(acc, 4))

# ---------------------------
# 9. 시각화
# ---------------------------
# 이진 분류이므로 np.argmax 제거
images = x_test  # 이미 250x250x3 형태

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i])
    plt.title(f"예측:{int(y_pred_binary[i][0])} / 정답:{int(y_test[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# loss :  0.6342740058898926
# acc :  0.6465256810188293
# 걸린시간: 166.06285858154297 초