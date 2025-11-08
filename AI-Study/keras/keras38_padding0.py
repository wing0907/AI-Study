from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D


# 2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1),
                 strides=1,
                 padding='same',  # padding에 대한 값 = same, valid.  same은 shape 유지
                # padding='valid',
                 ))
model.add(Conv2D(filters=9, kernel_size=(3,3),
                 strides=1,
                 padding='valid'  # 디폴트가 valid 임
                 ))
model.add(Conv2D(8, 4)) # 4 =  4,4 로 인식함 2라고 쓰면 2,2로 인식함

model.summary()

 # padding='valid',
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 9, 9, 10)          50

#  conv2d_1 (Conv2D)           (None, 7, 7, 9)           819

#  conv2d_2 (Conv2D)           (None, 4, 4, 8)           1160

# =================================================================
# Total params: 2,029
# Trainable params: 2,029
# Non-trainable params: 0


# padding='same'
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 10, 10, 10)        50

#  conv2d_1 (Conv2D)           (None, 8, 8, 9)           819

#  conv2d_2 (Conv2D)           (None, 5, 5, 8)           1160

# =================================================================
# Total params: 2,029
# Trainable params: 2,029
# Non-trainable params: 0