import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 인스턴스를 정의한다 클래스를 만들어서 변수를 정의한다. 고가시가 인스턴스화 한다.
# 이미지를 가져다가 직접 수치화 하는 함수쓰
train_datagen = ImageDataGenerator(
    rescale=1./255,         #정규화 0~255 scaling.
    horizontal_flip=True,   #수평반전  <- 데이터 증폭 또는 변환
    vertical_flip=True,     #수직반전 <- 데이터 증폭 또는 변환
    width_shift_range=0.1,  #수치를 10% 이동 하겠다. 평행이동 10%
    height_shift_range=0.1, #수직이동 10%
    rotation_range=5,       #360도 중 5도를 돌린다
    zoom_range=1.2,         #1.2배 확대
    shear_range=0.7,        #좌표점 고정 후 일부를 한쪽으로 끌어당기는 범위(찌부 만들기)
    fill_mode='nearest'     
    
)

test_datagen = ImageDataGenerator(      #test에는 증폭 또는 변환 시키지 않는다. 평가 데이터는 수정하지 않는다.
    rescale=1./255,
)


path_train = 'C:\Study25\_data\image\\brain\\train\\'
path_test = 'C:\Study25\_data\image\\brain\\test\\'

xy_train = train_datagen.flow_from_directory(
    path_train,                    #경로
    target_size=(200, 200),        #원 사이즈 150,150 가 해당 사이즈로 자동변환 됨 resize
    batch_size=10,                #메모리 사이즈 때문에 batch를 통배치로 하지 않는다
    class_mode='binary',           #이진분류
    color_mode='grayscale',        #색깔
    shuffle=True,
    seed=333,
)
# Found 160 images belonging to 2 classes. 

xy_test = test_datagen.flow_from_directory(
    path_test,                     #경로
    target_size=(200, 200),        #원 사이즈 150,150 가 해당 사이즈로 자동변환 됨 resize
    batch_size=10,                 #
    class_mode='binary',           #이진분류
    color_mode='grayscale',        #색깔
    # shuffle=True, 평가라서 shuffle 할 필요없음
)
# Found 120 images belonging to 2 classes.
print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000018F7F4190A0>
print(xy_train[0]) #이 데이터셋의 0번째 보여줘
# array([0., 0., 1., 1., 1., 0., 1., 1., 1., 0.] batch_size=10 ad와 normal의 두개 폴더에서 무작위(shuffle)로 10개 지정한 것
print(len(xy_train))
# 16. (0부터 15까지 들어있음. batch가 10 이니까.)
print(xy_train[0][0].shape)     # (10, 200, 200, 1)
print(xy_train[0][1].shape)     # (10,)
print(xy_train[0][1])           # [0. 0. 1. 1. 1. 1. 0. 0. 0. 0.]

# 에러코드
# print(xy_train[0].shape)      # AttributeError: 'tuple' object has no attribute 'shape'
# print(xy_train[16])           # ValueError: Asked to retrieve element 16, but the Sequence has length 16 (0부터 15까지라서 에러)
# print(xy_train[0][2])         # IndexError: tuple index out of range

print(type(xy_train))       # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'> 파이썬의 기초적인 데이터 형태. 1.리스트(대괄호, 수정가능) 2.튜플(소괄호, 수정불능) 3.딕셔너리(키 밸류) 4.SET 잘쓰지않음
print(type(xy_train[0][0])) # <class 'numpy.ndarray'> = 0번째 배치의 x data 
print(type(xy_train[0][1])) # <class 'numpy.ndarray'> = 0번째 배치의 y data 



#20250613
# Class = 붕어빵 틀 이다. 이걸 보고 붕어빵이라고 하지 않는다. 붕어빵틀은 붕어빵으로 실존하지 않음
# '내 붕어빵' 이라는 실물을 만드는 것. 실존화 시키는 것 = 인스턴스화. 
# 붕어빵이라는 클래스에 파라미터에 팥이 있을 수 있고, 이름도 팥 붕어빵이라고 저장할 수 있고
# 내용물을 바꿀 수 도 있는게 인스턴스화(실체화)이다.
# '클래스' 만으로는 실행를 할 수가 없기 때문에 객체 (인스턴스)화 해서 실행 하는 것임.

 # 변수 (지금까지 배운 느낌으로는 저장소의 너낌쓰..)
# x = train_csv.drop(['attack_type','ip_src', 'ip_dst'], axis=1)
# y = train_csv['attack_type']

