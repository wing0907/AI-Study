import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=10.0, size=1000)
# 지수분포의 평균(mean) 2.0
print(data)
print(data.shape)       # (1000,)
print(np.min(data), np.max(data))
# 0.0018484869314875166 13.900060381114514

log_data = np.log1p(data)       # np.expm1p(data) log1p의 반대 
                
                # 데이터에다 log를 씌운것 뿐이야. 얘가 있단 말이야. 으이!?!?! 으이!?!?!?!?!?!?

# np.log(data) # 왜 나는 np.log1p를 썼을까!? 으이!?!? 틀렸는데 정윤이 다시 얘기해봐.
             # 아이삭 알겠늬이??! 로그에 로그에 뭐야. 로그 0이면 몇이야. 자, 로그 100이면?
             # 로그 0은 없지?! 만약에
             # 이 중에 하나라도 0이면 큰일은 아니지만 에러가 나거나 하다못해 warning이라도 나오겠지?!

plt.subplot(1,2,1)             
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('Original')

plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='red', alpha=0.5)
plt.title('Log Transformed')
plt.show()

#  x 에서 가능하다! 대신 똑같은 위치가 되야한다.
#  y 에서 가능하다! 통상 로그변환은 y에서 더 많이 해! 맞!냐아아!!!!
#  어차피 오차부분은 못 맞춰..! 재현이 주식 예측을 생각해보셈.
#  앞쪽 전체적인 덩어리가 맞으면 될거아니야.
#  수치상으로는 1이나 2밖에 안되자나! linear로 더 잘 맞추겠지
#  너무 크면 log 변환 생각하고, 너무 크지도 않은데 log 변환 후 성능이 좋아질 수도 있다.
#  y 에 대한 스케일링 이라고 생각하면 됨!
#  준혁이 아픔..ㅜ.ㅜ 매우아픔...ㅜ.ㅜ
#  주의 할건 np.log1p를 해야 한다

