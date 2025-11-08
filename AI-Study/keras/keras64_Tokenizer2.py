import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

text1 = '오늘도 못생기고 영어를 디게 못 하는 이삭이는 재미없는 개그를 \
        마구 마구 마구 마구 하면서 딴짓을 한다.'     
#  어절 단위로 수치화한다.
text2 = '오늘도 박석사가 자아를 디게 디게 찾아냈다. 상진이는 마구 마구 딴짓을 한다. \
        재현은 못생기고 재미없는 딴짓을 한다.'

token = Tokenizer()
token.fit_on_texts([text1, text2])

print(token.word_index)
# {'마구': 1, '디게': 2, '딴짓을': 3, '한다': 4, '오늘도': 5,
#  '못생기고': 6, '재미없는': 7, '영어를': 8, '못': 9, '하는': 10,
#  '이삭이는': 11, '개그를': 12, '하면서': 13, '박석사가': 14, '자아를': 15,
#  '찾아냈다': 16, '상진이는': 17, '재현은': 18}

print(token.word_counts)
# OrderedDict([('오늘도', 2), ('못생기고', 2), ('영어를', 1), ('디게', 3),
#              ('못', 1), ('하는', 1), ('이삭이는', 1), ('재미없는', 2),
#              ('개그를', 1), ('마구', 6), ('하면서', 1), ('딴짓을', 3),
#              ('한다', 3), ('박석사가', 1), ('자아를', 1), ('찾아냈다', 1), ('상진이는', 1), ('재현은', 1)])
x = token.texts_to_sequences([text1, text2])
print(x)
# [[5, 6, 8, 2, 9, 10, 11, 7, 12, 1, 1, 1, 1, 13, 3, 4], [5, 14, 15, 2, 2, 16, 17, 1, 1, 3, 4, 18, 6, 7, 3, 4]]
x = np.array(x)
print(x)
# [[ 5  6  8  2  9 10 11  7 12  1  1  1  1 13  3  4]
#   [ 5 14 15  2  2 16 17  1  1  3  4 18  6  7  3  4]]
x = np.concatenate(x)
print(x)
# [ 5  6  8  2  9 10 11  7 12  1  1  1  1 13  3  4  5 14 15  2  2 16 17  1
#   1  3  4 18  6  7  3  4]
from sklearn.preprocessing import OneHotEncoder

# sklearn
# x = np.reshape(x, (-1, 1))
# print(x.shape)      # (32, 1)
# ohe = OneHotEncoder(sparse=False)          
# x = ohe.fit_transform(x)
# print(x.shape)      # (32, 18)

# pandas
# x = pd.get_dummies(np.array(x).reshape(-1, ))
# print(x)
# print(x.shape)        # (32, 18)

from tensorflow.keras.utils import to_categorical

# keras
# # 1. NumPy 배열로 변환 후 1 빼기
x_arr = np.array(x) - 1  # 중요: 1부터 시작한 index를 0부터로 조정

# # 2. One-hot 인코딩
x_onehot = to_categorical(x_arr)

# # 3. 차원 확인 및 reshape (필요 시)
print(x_onehot.shape)  # (32, 18) ← now correct

# # 4. (32, 18)으로 reshape
# x_onehot = x_onehot.reshape(-1, x_onehot.shape[-1])
# print(x_onehot.shape)  