from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D # 통상 4차원을 출력한다

# 5,5,1 짜리 이미지. (N, 5, 5, 1) 흑백이미지. 가로, 세로, 컬러 컬러이면 3
                                                # 세로,   가로,    색깔(color)
model = Sequential()                            # height, width, channels (지엄하신 분들과 얘기할 때ㅎㅎ)
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))    #(None, 4, 4, 10) 
model.add(Conv2D(5, (2,2)))     # (3, 3, 5)  # kernel : 가중치
model.add(Conv2D(5, (2,2)))     # (2, 2, 5)

""" keras.io -> API DOCS 에서 CONV2D layer 들어가면 보이는 코딩구조
keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)"""
model.add(Flatten())        # (None, 20) # 2차원으로 한방에 바꿔주는 함수 # 연산량은 없다
model.add(Dense(units=10))  # input = (batch, input_dim) (None, 10)
model.add(Dense(3))         # Dense: 2차원 입력에 2차원 출력이지만 다차원 입력 다차원 출력이 됨
# (None, 3)                 # Dense를 사용하려면 reshape 하면 된다
                            # Flatten 사용하면 됨

""" keras.io -> API DOCS -> core layer에서 확인 가능
keras.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    lora_rank=None,
    lora_alpha=None,
    **kwargs
)"""



model.summary()