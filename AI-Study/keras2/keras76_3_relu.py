import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

def relu(x):
    return np.maximum(0, x) # relu는 0과 비교하면 끝

# relu = lambda x : np.maximum(0, x) # lambda에 relu적용. # lambda 또는 함수 둘 중 하나 사용

y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# 한정 함수 또는 활성화 함수 라고 함