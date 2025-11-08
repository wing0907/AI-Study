# copy from 47

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img       # 이미지 땡겨오기.
from tensorflow.keras.preprocessing.image import img_to_array   # 이미지를 수치화 시킨다
import matplotlib.pyplot as plt
import numpy as np

path = 'c:/Study25/_data/image/me/'

img = load_img(path + 'prettyiu.jpg', target_size=(250, 250), )
print(img)      # <PIL.Image.Image image mode=RGB size=250x250 at 0x233683689D0>
print(type(img))    # <class 'PIL.Image.Image'>
# PIL = Python Image Library

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)        # (250, 250, 5)
print(type(arr))        # <class 'numpy.ndarray'>

### 3차원 -> 4차원 바꿔주기 (차원증가) ###
# arr = arr.reshape(1, 250, 250, 3)
# print(arr)
# print(arr.shape)        # (1, 250, 250, 3)

img = np.expand_dims(arr, axis=0) # 축 0번째, 현재는 1임
print(img.shape)          # (1, 250, 250, 3) expad_dims는 1,를 어디다 넣을거냐 이다. 2넣을거면 reshape해야함

# me 폴더에 데이터를 npy로 저장하겠다
# np.save(path + 'keras47_me_2.npy', arr=img)


################# 요기부터 증폭 ######################

datagen = ImageDataGenerator(
    rescale=1./255,         #정규화 0~255 scaling.
    # horizontal_flip=True,   #수평반전  <- 데이터 증폭 또는 변환 /  좌우반전
    vertical_flip=True,     #수직반전 <- 데이터 증폭 또는 변환 / 상하반전
    # width_shift_range=0.1,  #수치를 10% 이동 하겠다. 평행이동 10%
    height_shift_range=0.1, #수직이동 10%
    # rotation_range=15,       #360도 중 n도를 돌린다
    # zoom_range=1.2,         #1.2배 확대
    # shear_range=0.7,        #좌표점 고정 후 일부를 한쪽으로 끌어당기는 범위(찌부 만들기)
    fill_mode='nearest'     
    
)

it = datagen.flow(img,      #수치화된 데이터 사용하기 때문에 경로 필요 없음. resize 할 거 아니면 size도 필요 없음.
    batch_size=1,           #결국엔 batch_size만 남음.
)

print("====================================================")
print(it)       # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001EA73A32CD0>
print("====================================================")
# iterator 라는 놈을 next라는 함수에 집어넣으면 그 첫번째 다음 두번째, 순서대로 나옴.
# print(it.next())

# aaa = it.next()   # python 2.0 문법
# print(aaa)
# print(aaa.shape)  # (1, 250, 250, 3)

# bbb = next(it)      #현재 문법        # iterator 나오면 2가지 생각하면 됨. 1. for(), 2. next()
# print(bbb)          # next 쓰면 iterator의 다음거.  
# print(bbb.shape)    # (1, 250, 250, 3)

# print(it.next())      #원래는 에러남. NumpyArrayIterator 요놈이 예외라서 출력되는 거임.
# print(it.next())      #보통 iterator의 전체개수 이상만큼 next 돌리면 에러남
# print(it.next())

########## RAG 로 만든 3*5 짜리 코드 ###########
fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(10,5))
# for i in range(10): # 우리는 5번 돌릴거야.
#     # batch = it.next()     # IDG에서 랜덤으로 한 번 작업 (변환)
#     batch = next(it)
#     print(batch.shape)      #(1, 250, 250, 3)
#     batch = batch.reshape(250, 250, 3)
#     if i < 5:
#         ax[0][i].imshow(batch)
#         ax[0][i].axis('off')
#     else:
#         ax[1][i-5].imshow(batch)
#         ax[1][i-5].axis('off')


# for i in range(50):
#     batch = next(it).reshape(250, 250, 3)
#     row, col = divmod(i, 10)
#     ax[row][col].imshow(batch)
#     ax[row][col].axis('off')

    # ax[i].imshow(batch)
    # ax[i].axis('off')

ax = ax.flatten()  # 2차원을 1차원으로 변경
# ax = ax.ravel() 

for i in range(15):
    batch = next(it)                 # (1, 250, 250, 3)
    batch = batch.reshape(250, 250, 3)

    ax[i].imshow(batch)
    ax[i].axis('off')
plt.tight_layout        # 겹치지 않도록 레이아웃을 만들어 줌
plt.show()

