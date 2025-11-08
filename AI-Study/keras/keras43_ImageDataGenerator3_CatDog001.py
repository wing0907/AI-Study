import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import os

# 1. 데이터 경로
path = 'C:/Study25/_data/kaggle/cat_dog/'
path_train = path + 'train2/'
path_test = path + 'test2/'
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 3. Generator 정의
train_generator = train_datagen.flow_from_directory(
    path_train,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    subset='training',
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    path_train,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    subset='validation',
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    path_test,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    color_mode='rgb',
    
)



# 4. 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    BatchNormalization(),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# 5. 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. 콜백 설정
os.makedirs('./_save/cat_dog/', exist_ok=True)
mcp = ModelCheckpoint(
    filepath='./_save/cat_dog/catdog_best_{val_loss:.4f}.hdf5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

# 7. 학습
start = time.time()
history = model.fit(
    train_generator,
    epochs=2,
    validation_data=val_generator,
    callbacks=[es, mcp],
    verbose=1
)
end = time.time()



# 8. 평가
loss, acc = model.evaluate(test_generator)
print('✅ Test Loss:', round(loss, 4))
print('✅ Test Accuracy:', round(acc, 4))
print('⏱️ 걸린 시간:', round(end - start, 2), '초')

# ✅ Test Loss: 1.119
# ✅ Test Accuracy: 0.4725
# ⏱️ 걸린 시간: 1359.2 초

test_generator.reset()
y_pred_prob = model.predict(test_generator, verbose=1)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)

# ✅ 실제 라벨
y_true = test_generator.classes

# ✅ 메모리 절약을 위해 25개만 로드
x_images = []
for i in range((25 // test_generator.batch_size) + 1):  # 필요한 배치 수만큼만 반복
    x_batch, _ = test_generator[i]
    x_images.extend(x_batch)
    if len(x_images) >= 25:
        break

x_images = np.array(x_images[:25])  # 정확히 25개로 자르기

# ✅ 시각화
plt.figure(figsize=(16, 6))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(x_images[i])
    plt.title(f"예측:{y_pred[i]} / 정답:{int(y_true[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()

import datetime
import os
# 파일명에 날짜 포함
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission_filename = f"submission_result_{date}.csv"
submission_save_path = os.path.join(path, submission_filename)

# 파일명 정리
file_names = [os.path.basename(f) for f in test_generator.filenames]
submission_csv['Prediction'] = y_pred
submission_csv.index = file_names

# 저장
submission_csv.to_csv(submission_save_path)
print("📄 제출 파일 저장 완료:", submission_save_path)