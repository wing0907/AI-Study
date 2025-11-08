import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, BatchNormalization, MaxPooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


np_path = 'c:/study25/_data/_save_npy/'
# np.save(np_path + "keras44_01_x_train.npy", arr=x)
# np.save(np_path + "keras44_01_y_train.npy", arr=y)

start = time.time()
x1_train = np.load(np_path + "keras46_03_x_train.npy")
y1_train = np.load(np_path + "keras46_03_y_train.npy")
# x_test = np.load(np_path + "keras46_02_x_test.npy")
# y_test = np.load(np_path + "keras46_02_y_test.npy")
end = time.time()

print(x1_train)
print(y1_train[:20])    # [1. 0. 0. 0. 0. 1. 0. 0. 1. 1. 1. 0. 1. 0. 0. 1. 1. 0. 0. 1.]
print(x1_train.shape, y1_train.shape)  # (1027, 100, 100, 3) (1027,)
print("load time :", round(end-start, 2),"seconds")
#  load time : 33.87 seconds

x1_train = x1_train.reshape(-1, 100*100, 3)


x_train, x_test, y_train, y_test = train_test_split(
    x1_train, y1_train, test_size=0.3, random_state=333,                   
    shuffle=True, 
)

model = Sequential([
    LSTM(64, activation='relu', input_shape=(100*100, 3)),
    BatchNormalization(),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])


model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=40, verbose=1,
                   restore_best_weights=True,
                   )
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")   

path = './_save/horse_or_human/'
filename = '.hdf5'
filepath = "".join([path, 'k46_04',filename])
#######################################################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model.fit(x_train, y_train, epochs=5,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp])
end = time.time()

# 평가
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss : ', loss[0])
print('acc : ', loss[1])
print('걸린시간:', end - start, '초')

# 예측
y_pred = model.predict(x_test)

# y_submit = model.predict(x_test) 
# submission_csv['label'] = y_submit # 예측 결과 넣기
# import datetime
# date = datetime.datetime.now().strftime("%m%d_%H%M")
# submission_csv.to_csv(path + f'sample_submission_{date}.csv')


# # 시각화용 reshape
# images = x_test.reshape(-1, 100, 100, 3)

# # 시각화
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# matplotlib.rcParams['axes.unicode_minus'] = False 

# plt.figure(figsize=(16, 6))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.imshow(images[i], cmap='winter')
#     plt.title(f"예측:{y_pred[i]} / 정답:{y_test[i]}")
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

# loss :  0.0033516467083245516
# acc :  1.0
# 걸린시간: 80.49675488471985 초

# loss :  0.6947099566459656
# acc :  0.5113268494606018
# 걸린시간: 1870.906555891037 초