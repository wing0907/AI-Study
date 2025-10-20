import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


np_path = 'c:/study25/_data/_save_npy/'
# np.save(np_path + "keras44_01_x_train.npy", arr=x)
# np.save(np_path + "keras44_01_y_train.npy", arr=y)

start = time.time()
x_train = np.load(np_path + "keras44_01_x_train.npy")
y_train = np.load(np_path + "keras44_01_y_train.npy")
end = time.time()

print(x_train)
print(y_train[:20])
print(x_train.shape, y_train.shape)  # (25000, 200, 200, 3) (25000,)
print("load time :", round(end-start, 2),"seconds")
#  load time : 33.87 seconds


exit()

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


model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=30, verbose=1,
                   restore_best_weights=True,
                   )
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   

path = './_save/keras43/'
filename = '.hdf5'
filepath = "".join([path, 'k43_01',filename])
#######################################################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model.fit(xy_train, epochs=200, batch_size=4,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp])
end = time.time()

# 평가
loss = model.evaluate(xy_test, verbose=1)
print('loss : ', loss[0])
print('acc : ', loss[1])
x_test = xy_test[0][0]
y_test = xy_test[0][1]
# 예측
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int).flatten()  # (N, 1) → (N,)
y_test = y_test.flatten()                      # (N, 1) → (N,)

# 정확도
acc = accuracy_score(y_test, y_pred)
print('acc:', round(acc, 4))
print('걸린시간:', end - start, '초')

# 시각화용 reshape
images = x_test.reshape(-1, 200, 200)

# 시각화
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=(16, 6))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"예측:{y_pred[i]} / 정답:{y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# loss :  0.0348055437207222
# acc :  0.9916666746139526
# acc: 0.9917
# 걸린시간: 149.21810269355774 초


# loss :  0.02179562672972679
# acc :  1.0
# acc: 1.0
