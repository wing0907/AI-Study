import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

tanh = lambda x : 1 / (1 + np.exp(-np.tan(x)))  # tanh을 lambda로 구현


# 원래 정의    
sigmoid_tan = lambda x: 1 / (1 + np.exp(-np.tan(x)))    # 1번 형태: 분자에 지수항이 있어서 미분 시 계산이 편해짐

# e^t / (e^t + 1) 형태  
sigmoid_tan_alt = lambda x: np.exp(np.tan(x)) / (np.exp(np.tan(x)) + 1) # 2번 형태: tanh 이용하면 안정적이고 오버플로우를 줄임

# tanh 기반  
sigmoid_tan_tanh = lambda x: 0.5 * (1 + np.tanh(np.tan(x) / 2)) # 3번 형태: 코드 구현 선택지가 다양함



y = np.tanh(x)          # -1 에서 1 사이

plt.plot(x, y)
plt.grid()
plt.show()


# 한정 함수 또는 활성화 함수 라고 함