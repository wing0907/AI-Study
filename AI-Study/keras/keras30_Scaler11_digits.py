from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === Load Data ===
# dataset = load_digits()
# print(dataset)
# print(dataset.DESCR)
# print(dataset.feature_names)

# x = dataset.data
# y = dataset.target

# print(x)
# print(x.shape)      #  (1797, 64)
# print(y)
# print(y.shape)      #  (1797,)

# 1. 데이터 로드
digits = load_digits()
x, y = digits.data, digits.target

# x = np.min(x), np.max(x)
# print(x)  # (0.0, 16.0)

x = x.reshape(-1, 8, 8, 1)
import pandas as pd
from tensorflow.keras.layers import BatchNormalization, Dropout
digits = load_digits()
x = digits.images.reshape(-1, 8, 8, 1).astype("float32") / 16.0
y = pd.get_dummies(y)



# 2. 학습/테스트 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=digits.target, random_state=555
)


model = Sequential()                           
model.add(Conv2D(filters=32, kernel_size=(2,2), input_shape=(8,8,1), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))   
model.add(Conv2D(16, (2,2), activation='relu'))   
model.add(Flatten()) # 2차원으로 한방에 바꿔주는 함수
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))   
model.add(Dense(y.shape[1], activation='softmax'))

#3. 컴파일, 훈련
labels = np.argmax(y_train.values, axis=1)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(weights))

model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, 
          epochs=100,
          batch_size=4,
          validation_split=0.2,
          verbose=1,
          class_weight=class_weights
        )


loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)
y_pred = np.argmax(result, axis=1)
y_true = np.argmax(y_test.values, axis=1)
print('loss:', loss[0])
print('acc:', loss[1])
print('예측값: ', y_pred[:10])
print('실제값: ', y_true[:10])

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
matplotlib.rcParams['axes.unicode_minus'] = False 

images = x_test.reshape(-1, 8, 8)  

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"예측:{y_pred[i]} / 정답:{y_true[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

aaa = 5
print(y_train[aaa])

import matplotlib.pyplot as plt
# plt.imshow(x_train[1], 'gray')
plt.imshow(x_train[aaa], 'gray')
plt.show()



# cm = confusion_matrix(y_true, y_pred)
# cm_df = pd.DataFrame(cm, index=range(10), columns=range(10))
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
# plt.xlabel('Predicted Label', fontsize=12)
# plt.ylabel('True Label', fontsize=12)
# plt.title('📊 Confusion Matrix (Actual vs Predicted)', fontsize=14)
# plt.tight_layout()
# plt.show()

# loss: 0.10773900151252747
# acc: 0.9777777791023254
# 예측값:  [3 4 4 4 9 5 3 6 8 6]
# 실제값:  [3 4 4 4 9 5 3 6 8 6]