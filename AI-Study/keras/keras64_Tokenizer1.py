import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

text = '오늘도 못생기고 영어를 디게 못 하는 이삭이는 재미없는 개그를 \
        마구 마구 마구 마구 하면서 딴짓을 한다.'     
#  어절 단위로 수치화한다.

token = Tokenizer() # 객채, 변수  // 클래스
token.fit_on_texts([text]) # 문장이 여러개여서 리스트형태

print(token.word_index)
# {'마구': 1, '오늘도': 2, '못생기고': 3, '영어를': 4, '디게': 5, '못': 6,
#  '하는': 7, '이삭이는': 8, '재미없는': 9,
#  '개그를': 10, '하면서': 11, '딴짓을': 12, '한다': 13}

print(token.word_counts)
# OrderedDict([('오늘도', 1), ('못생기고', 1), ('영어를', 1), ('디게', 1),
#              ('못', 1), ('하는', 1), ('이삭이는', 1), ('재미없는', 1),
#              ('개그를', 1), ('마구', 4), ('하면서', 1), ('딴짓을', 1), ('한다', 1)])

x = token.texts_to_sequences([text])
print(x)
# [[2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 1, 1, 1, 11, 12, 13]]
# 수치화 한 데이터가 맞는지 항상 double check!
# 리니어 모델이라면 값이 상관없지만 이럴 경우엔 onehotenconding 해야 함



############## 원핫 3가지 만들기 ################
#===================================================================#

# 1. sklearn - OneHotEncoder              재익은 재현의 두배가 아니야.   # 다중분류에서는 "반드시" OneHotEncoding을 사용하고 Y 만 적용한다!!!!!!!!

from sklearn.preprocessing import OneHotEncoder
import numpy as np

# # 쌤
# x = np.reshape(x, (-1, 1))
# print(x.shape)      # (16, 1)
# ohe = OneHotEncoder(sparse=False)          # sparse_output=Fasle
# x = ohe.fit_transform(x)
# print(x.shape)      # (16, 13)

# exit()



# x는 2차원 리스트 [[2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 1, 1, 1, 11, 12, 13]]
# 먼저 flatten 하고 reshape 해줘야 함

# x_array = np.array(x).flatten()      # (1, 16) → (16,)
# x_reshaped = x_array.reshape(-1, 1)  # (16,) → (16, 1)

# encoder = OneHotEncoder(sparse=False)
# x_onehot = encoder.fit_transform(x_reshaped)

# print(x_onehot)
# print(x_onehot.shape)       # (16, 13)


# 장점: 데이터가 2D 배열이어야 하고, 범주형 데이터를 원-핫 벡터로 변환할 때 많이 씀.
# sparse=False가 없으면 희소 행렬 반환.
#####################################################################
# from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(-1, 1)                # reshape 바뀌지 않아야 할것 2가지 =  1. 안에 들어간 값  / 2. 순서
# ohe = OneHotEncoder()               # metrics형태를 받기때문에 (N, 1)로 reshape하고 해야한다.
# y = ohe.fit_transform(y)#.toarray()  # .toarray()를 쓰면 numpy 형태로 바뀜
# print(y)                            # 희소행렬방식
# print(y.shape)   # (150, 3)
# print(type(y))                      # <class 'scipy.sparse.csr.csr_matrix'>
# y = y.toarray()                     # scipy를 numpy로 변환
# ohe = OneHotEncoder(sparse=False)   # numpy형태 출력, 디폴트는 True
# y = ohe.fit_transform(y)
# print(y)
#####################################################################
#===================================================================#
# 2. pandas - get_dummies
# import pandas as pd

# # 쌤
# x = pd.get_dummies(np.array(x).reshape(-1, ))
# print(x)
# print(x.shape)      # (16, 13)

# exit()


# # 1차원으로 변환
# x_flat = np.array(x).flatten()

# # pandas의 get_dummies 사용
# x_onehot = pd.get_dummies(x_flat)
# # 결과 확인
# print(x_onehot)
# print(x_onehot.shape)       # (16, 13)


# 장점: 데이터가 Series 형태일 때 바로 사용 가능.
# 결과는 DataFrame 형태.
# 쉽게 원-핫 인코딩이 가능하고, 다시 NumPy 배열로 바꿀 수도 있음: y_onehot.values
#===================================================================#

# 3. Keras - to_categorical                           0이 아닌 1부터 시작하는 데이터는 0이라는 컬럼이 새로 생겨버린다.
                                                    # 그래서 항상 찍어보고 .shape 확인해보기
from tensorflow.keras.utils import to_categorical
import numpy as np

# # 쌤
# x = to_categorical(x, num_classes=16)
# x = x[:, :, 1:]
# print(x.shape)      # (1, 16, 15)
# x = x.reshape (16, 13)
# print(x.shape)

# exit()


x_onehot = to_categorical(x)
print(x_onehot)
print(x_onehot.shape)       # (1, 16, 14)


# 원본: [[2, 3, 4, ..., 13]]
# x = [[2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 1, 1, 1, 11, 12, 13]]

# # 1. NumPy 배열로 변환 후 1 빼기
# x_arr = np.array(x) - 1  # 중요: 1부터 시작한 index를 0부터로 조정

# # 2. One-hot 인코딩
# x_onehot = to_categorical(x_arr)

# # 3. 차원 확인 및 reshape (필요 시)
# print(x_onehot.shape)  # (1, 16, 13) ← now correct

# # 4. (16, 13)으로 reshape
# x_onehot = x_onehot.reshape(-1, x_onehot.shape[-1])
# print(x_onehot.shape)  # (16, 13)



# 장점: 정수 레이블 배열을 바로 받아서 원-핫 인코딩.
# TensorFlow/Keras 모델 학습 시 가장 간단하고 직관적.

#===================================================================#

# x는 다음과 같은 형태야:
# x = [[2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 1, 1, 1, 11, 12, 13]]  # shape: (1, 16)
# 즉, 2차원 리스트 (배치 크기 1, 시퀀스 길이 16)

# 이걸 to_categorical()에 넣으면:
# 각 값마다 원-핫 벡터로 바꾸고
# 전체 시퀀스를 유지
# 따라서 shape: (batch_size, sequence_length, num_classes)

# 결과적으로:
# (1, 16, 14)
# # 1: 문장(시퀀스) 개수
# 16: 어절 수
# 14: 고유 토큰 개수 (index 0 ~ 13 → 총 14개)


# to_categorical()은 기본적으로 0부터 시작하는 class index 기준으로 one-hot을 만드는데,
# Tokenizer의 word_index는 1부터 시작.
# 즉, 1 → [0, 1, 0, 0, ..., 0], 2 → [0, 0, 1, 0, ..., 0] 이런 식으로 되고,
# index 0도 포함되기 때문에 총 클래스 수는 +1 되는 것.

# 예시
# x = [1, 2, 3]
# to_categorical(x) →
# [
#  [0, 1, 0, 0],
#  [0, 0, 1, 0],
#  [0, 0, 0, 1]
# ]

# 최대 값이 13 → class 개수는 14

# shape: (1, 16, 14)