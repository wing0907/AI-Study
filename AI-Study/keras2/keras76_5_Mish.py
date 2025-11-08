import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x))) # relu에 비해 단점이 있다. 데이터가 커지면 연산량이 많아서 오래걸림  
                                              # e의 x승만큼 연산이 되기때문.
                                              
# mish = lambda x: x * np.tanh(np.log1p(np.exp(x)))  # log1p는 log(1+exp(x))의 안정적 버전
                                                   # lambda 또는 함수 둘 중 하나 사용
y = mish(x)

plt.plot(x, y)
plt.grid()
plt.show()

# 한정 함수 또는 활성화 함수 라고 함