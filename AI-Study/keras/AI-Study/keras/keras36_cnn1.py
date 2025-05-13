from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D # 통상 4차원을 출력한다

# 5,5,1 짜리 이미지. (N, 5, 5, 1) 흑백이미지. 가로, 세로, 컬러 컬러이면 3
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1)))       #(None, 4, 4, 10) 
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          50

# =================================================================
# Total params: 50
# Trainable params: 50
# Non-trainable params: 0
model.add(Conv2D(5, (2,2)))     # (3, 3, 5)
model.add(Conv2D(5, (2,2)))     # (2, 2, 5)


model.summary()