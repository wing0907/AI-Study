import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def silu(x):
    return x * (1 / (1 + np.exp(-x))) # 2017년도 구글에서 swish라고 발표.  
                                      # 음수 구간에서도 완전히 0이 되지 않아서 Gradient Dead Zone 문제를 줄임
                                      # 최근 딥러닝 모델들(예: EfficientNet)에서 자주 사용됨
# silu = lambda x: x / (1 + np.exp(-x)) # lambda 또는 함수 둘 중 하나 사용

y = silu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# 한정 함수 또는 활성화 함수 라고 함