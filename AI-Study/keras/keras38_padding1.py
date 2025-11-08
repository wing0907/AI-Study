# 100, 100, 3 이미지를 10, 10, 11 으로 줄이기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D


# 2. 모델구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(100,100,3),
                 strides=1,
                 padding='same',  # padding에 대한 값 = same, valid.  same은 shape 유지
                # padding='valid',
                 ))
model.add(MaxPooling2D())                                              
model.add(MaxPooling2D())                                            
model.add(MaxPooling2D())                                               
model.add(Conv2D(filters=11, kernel_size=(2,2),
                 strides=1,
                 padding='valid'  # 디폴트가 valid 임
                 ))
model.add(Conv2D(11, 2)) 

model.summary()

#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 100, 100, 10)      130

#  max_pooling2d (MaxPooling2D  (None, 50, 50, 10)       0
#  )

#  max_pooling2d_1 (MaxPooling  (None, 25, 25, 10)       0
#  2D)

#  max_pooling2d_2 (MaxPooling  (None, 12, 12, 10)       0
#  2D)

#  conv2d_1 (Conv2D)           (None, 11, 11, 11)        451

#  conv2d_2 (Conv2D)           (None, 10, 10, 11)        495

# =================================================================
# Total params: 1,076
# Trainable params: 1,076
# Non-trainable params: 0