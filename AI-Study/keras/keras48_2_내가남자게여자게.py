import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img       # 이미지 땡겨오기.
from tensorflow.keras.preprocessing.image import img_to_array   # 이미지를 수치화 시킨다
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight


np_path = 'c:/Study25/_data/image/me/'

x_pred = np.load(np_path + "keras47_me_1.npy")  
y_test = [0]
path = 'C:\Study25\_save\horse_or_human\\'

model = load_model(path + 'k46_08.hdf5')

x_pred = x_pred.astype(np.float32) / 255.0


model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['acc'])


# 평가
y_predict = model.predict(x_pred)
y_predict = (y_predict > 0.5).astype(int)


accuracy_score = accuracy_score(y_test, y_predict)
print('acc:', accuracy_score)

# # 시각화용 reshape
images = x_pred.reshape(-1, 250, 250, 3)

# 시각화
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=(9, 6))
for i in range(len(x_pred)):  # x_pred 개수만큼 반복
    plt.subplot(1, len(x_pred), i+1)

    pred_class = y_predict[i][0] # 0 또는 1
    plt.imshow(x_pred[i])
    pred_label = '여자(1)' if y_predict[i][0] == 1 else '남자(0)'
    plt.title(f"예측: {pred_label} ")
    plt.axis('off')

    
plt.tight_layout()
plt.show()

