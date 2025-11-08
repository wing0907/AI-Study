import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])

def outlier(data):
    quartile_1, q2, quartile_3 = np.percentile(data, [25,50,75])
    print('1사분위 : ', quartile_1)
    print('2사분위 : ', q2)
    print('3사분위 : ', quartile_3)
    iqr = quartile_3 - quartile_1
    print('IQR : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)  # 왜 1.5인가? = 표준정규분포를 따랐을 때 나오는 수치가 1.5 임.
    return np.where((data > upper_bound) | (data < lower_bound)), \
        iqr, lower_bound, upper_bound   # | = 또는

outlier_loc, iqr, low, up = outlier(aaa)
print('이상치의 위치 : ', outlier_loc)
# 1사분위 :  4.0
# 2사분위 :  7.0
# 3사분위 :  10.0
# IQR :  6.0
# 이상치의 위치 :  (array([ 0, 12], dtype=int64),)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.axhline(up, color='brown', label='upper_bound')
plt.axhline(low, color='darkgreen', label='lower_bound')
plt.legend()
plt.show()