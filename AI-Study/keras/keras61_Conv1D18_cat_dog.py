import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt
import matplotlib

path = 'C:/Study25/_data/kaggle/cat_dog/'
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

np_path = 'c:/study25/_data/_save_npy/'

start = time.time()
x_train = np.load(np_path + "keras44_01_x_train.npy")
y_train = np.load(np_path + "keras44_01_y_train.npy")
x_test = np.load(np_path + "keras44_01_x_test.npy")
y_test = np.load(np_path + "keras44_01_y_test.npy")
end = time.time()

print("x_train shape:", x_train.shape)
print("y_train[:20]:", y_train[:20])
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print("load time:", round(end - start, 2), "seconds")

# reshape for LSTM: (samples, timesteps, features)
x_train = x_train.reshape(25000, -1)  # (25000, 120000)
x_test = x_test.reshape(12500, -1)    # (12500, 120000)


print(x_train.shape, x_test.shape)

# reshape to (timesteps=400, features=300)
x_train = x_train.reshape(-1, 50*50, 3)
x_test = x_test.reshape(-1, 50*50, 3)

x11_train, x11_test, y11_train, y11_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=190, shuffle=True)

model = Sequential([
    Conv1D(filters=64, kernel_size=2, 
           padding='same', activation='relu', input_shape=(50*50, 3)),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(8, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
date = datetime.datetime.now().strftime("%m%d_%H%M")
filepath = f'./_save/cat_dog/k45_{date}.hdf5'
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=filepath)

start = time.time()
hist = model.fit(x11_train, y11_train, epochs=1, validation_split=0.2, callbacks=[es, mcp])
end = time.time()

loss = model.evaluate(x11_test, y11_test, verbose=1)
print('loss : ', loss[0])
print('acc : ', loss[1])
print('걸린시간:', end - start, '초')

y_submit = model.predict(x_test)
submission_csv['label'] = y_submit
submission_csv.to_csv(path + f'sample_submission_{date}.csv')

# 시각화용 이미지 축소
# images = x_test.reshape(-1, 50*50, 3)

# plt.figure(figsize=(16, 6))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.imshow(images[i].reshape(20, 20, 3))  # 예시로 20x20으로 줄여서 시각화
#     plt.title(f"예측:{round(float(y_submit[i]),2)}")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()


# loss :  nan
# acc :  0.4946666657924652
# 걸린시간: 302.08364248275757 초


# loss :  0.6352799534797668
# acc :  0.6469333171844482
# 걸린시간: 10.798173904418945 초