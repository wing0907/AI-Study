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
np.save(path + 'keras47_me_2.npy', arr=img)



