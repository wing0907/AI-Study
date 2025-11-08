import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# def elu(x, alpha):
#     return (x>0)*x + (x<0)*(alpha*(np.exp(x)-1)) # elu는 -1에 수렴됨

# (x > 0) * x → x가 양수면 그대로 반환, 아니면 0
# (x <= 0) * (...) → x가 음수면 𝛼(𝑒𝑥−1)α(e−1) 반환, 아니면 0                                              
                                              
# elu = lambda x, alpha : (x>0)*x + (x<0)*(alpha*(np.exp(x)-1))
elu = lambda x, alpha: np.where(x > 0, x, alpha * (np.exp(x) - 1)) # 이렇게 구현도 가능. 가독성 better
                                                  
y = elu(x, alpha=-100) # 조절 가능.

plt.plot(x, y)
plt.grid()
plt.show()

# 한정 함수 또는 활성화 함수 라고 함