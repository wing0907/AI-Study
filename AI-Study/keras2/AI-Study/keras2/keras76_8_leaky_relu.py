import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

alpha = 0.01
def leaky_relu(x):
    # return np.maximum(alpha*x, x) # 0대신에 0.01과 비교하게 된다
    return np.where(x > 0, x, alpha * x) # np.where 사용 예시

# leaky_relu = lambda x : np.where(x > 0, x, alpha * x) # lambda에 leaky_relu적용. # lambda 또는 함수 둘 중 하나 사용

y = leaky_relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# 한정 함수 또는 활성화 함수 라고 함