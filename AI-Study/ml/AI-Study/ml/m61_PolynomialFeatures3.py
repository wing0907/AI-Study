# 선형 -> 비선형 효과적

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4,3)
print(x)
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]


pf = PolynomialFeatures(degree=2, include_bias=False,  # 다항식, default = True // degree=3 이면 3차원
                        interaction_only=True)         # 제곱이 빠지기 때문에 성능이 애매해짐
x_pf = pf.fit_transform(x)                             # 컬럼이 아주 적을때는 사용함. 컬럼이 아주 많을 때는 PCA로 압축을 때리던가 FI 나 Corr 로 가지치기 해야함
print(x_pf)                                            # 자잘한건 잘먹히지만 큰거는 잘 안먹힘
# [[  0.   1.   2.   0.   0.   2.]
#  [  3.   4.   5.  12.  15.  20.]
#  [  6.   7.   8.  42.  48.  56.]
#  [  9.  10.  11.  90.  99. 110.]]

