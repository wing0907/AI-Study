import numpy as np
from sklearn.preprocessing import StandardScaler

#1. 데이터
data = np.array([[1,2,3,1],
                [4,5,6,2],
                [7,8,9,3],
                [10,11,12,114],
                [13,14,15,115],]) # (5,4)
print(data.shape) # (5, 4)

#1) 평균
means = np.mean(data, axis=0)
print('평균 : ', means) # 평균 :  [ 7.  8.  9. 47.]

#2) 모집단 분산 (n으로 나눈다)      전체를 놓고 계산
population_variances = np.var(data, axis=0)
print('모집단 분산 : ', population_variances) 
# 모집단 분산 :  [  18.   18.   18. 3038.]


#3) 표본 분산 (n-1로 나눈다) 요만큼만 나눠서 한번 하고 요만큼만 나눠서 한번한거를 다 더해서 계산하는 것 (ex. 사전여론조사)
variances = np.var(data, axis=0, ddof=1)       # ddof : n-1빵 하겠다.
print('표본분산 : ', variances)     
# 표본분산 :  [  22.5   22.5   22.5 3797.5]


#4) 표본 표준편차  (통상적으로 표본이 더 작다)
stdl = np.std(data, axis=0, ddof=1)
print('표본 표준편차 : ', stdl)     
# 표본 표준편차 :  [ 4.74341649  4.74341649  4.74341649 61.62385902]

#5) 모집단 표준편차
std2 = np.std(data, axis=0)
print('모집단 표준편차 : ', std2)
# 모집단 표준편차 :  [ 4.24264069  4.24264069  4.24264069 55.11805512]

#6) StandardScaler (모집단) "표준 정규 분포로 의해 ~~"
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data) 
print('StandardScaler : \n', scaled_data)        # \n  : 다음 줄에 출력하겠다
""" <평균을 빼고 표편(모집단표편)으로 계산한 값> ex) 1-7 = -6 / 4.24264069 (통상 표준편차라고 하는 것은 모집단이다)
StandardScaler :
 [[-1.41421356 -1.41421356 -1.41421356 -0.83457226]
 [-0.70710678 -0.70710678 -0.70710678 -0.81642939]
 [ 0.          0.          0.         -0.79828651]
 [ 0.70710678  0.70710678  0.70710678  1.21557264]
 [ 1.41421356  1.41421356  1.41421356  1.23371552]]
"""
