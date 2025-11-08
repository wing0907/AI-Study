import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#  실무상에서는 폴더에서 이미지를 직접 수치화 해야 한다

# print(x_train)
# print(x_train[0])
# print(y_train[0]) # 5
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
    #   dtype=int64))
print(pd.value_counts(y_test))
# 1    1135
# 2    1032
# 7    1028
# 3    1010
# 9    1009
# 4     982
# 0     980
# 8     974
# 6     958
# 5     892

aaa = 7
print(y_train[aaa])

import matplotlib.pyplot as plt
# plt.imshow(x_train[1], 'gray')
plt.imshow(x_train[aaa])
plt.show()

# 이미지 데이터는 원래 shape (8, 8)이므로 다시 꺼내기
images = x_test.reshape(-1, 8, 8)

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"예측:{x_train[i]} / 정답:{y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()