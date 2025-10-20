import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

path = './_data/kaggle/cat_dog/'
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)  # 다시 읽기


np_path = 'c:/study25/_data/_save_npy/'
# np.save(np_path + "keras44_01_x_train.npy", arr=x)
# np.save(np_path + "keras44_01_y_train.npy", arr=y)

start = time.time()
x_train = np.load(np_path + "keras44_01_x_train.npy")
y_train = np.load(np_path + "keras44_01_y_train.npy")
x_test = np.load(np_path + "keras44_01_x_test.npy")
y_test = np.load(np_path + "keras44_01_y_test.npy")
end = time.time()

print(x_train)
print(y_train[:20])
print(x_train.shape, y_train.shape)  # (25000, 200, 200, 3) (25000,)
print(x_test.shape, y_test.shape)   #(12500, 200, 200, 3) (12500,)
print("load time :", round(end-start, 2),"seconds")
#  load time : 33.87 seconds


x11_train, x11_test, y11_train, y11_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=190,                   
    shuffle=True, 
)

model = Sequential([
    Conv2D(64, (2, 2), activation='relu', input_shape=(50, 50, 3)),
    BatchNormalization(),
    Dropout(0.3),
    MaxPooling2D(),
    Conv2D(16, (2, 2), activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(8, activation='relu'),
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

path = './_save/cat_dog/'
filename = '.hdf5'
filepath = "".join([path, 'k45_99',filename])
#######################################################
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath=filepath
)

start = time.time()
hist = model.fit(x11_train, y11_train, epochs=500,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[es, mcp])
end = time.time()

# 평가
loss = model.evaluate(x11_test, y11_test, verbose=1)
print('loss : ', loss[0])
print('acc : ', loss[1])
print('걸린시간:', end - start, '초')

# 예측
y_pred = model.predict(x11_test)

y_submit = model.predict(x_test) 
submission_csv['label'] = y_submit # 예측 결과 넣기
import datetime
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission_csv.to_csv(path + f'sample_submission_{date}.csv')


# # 시각화용 reshape
images = x_test.reshape(-1, 50, 50, 3)

# 시각화
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=(16, 6))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(images[i], cmap='winter')
    plt.title(f"예측:{y_pred[i]} / 정답:{y11_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 'k45_99'
# loss :  0.4635821282863617
# acc :  0.7901333570480347
# 걸린시간: 707.3447222709656 초