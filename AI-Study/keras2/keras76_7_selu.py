import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)

# SELU 고정 상수 (논문 제안값)
alpha = 1.6732632423543772
lmbda = 1.0507009873554805

def selu(x, alpha, lmbda):
    return lmbda * ((x > 0) * x + (x <= 0) * (alpha * (np.exp(x) - 1)))
                                              
selu = lambda x, alpha, lmbda : lmbda * ((x > 0) * x + (x <= 0) * (alpha * (np.exp(x) - 1))) 

# selu = lambda x: lmbda * ((x > 0) * x + (x <= 0) * (alpha * (np.exp(x) - 1)))
# 실제로는 SELU의 α 와 𝜆 값이 논문에서 고정 상수로 제안되었기 때문에,
# 대부분은 매번 alpha, lmbda 를 입력받기보다 고정값을 바로 넣어서 쓴다.

# selu = lambda x: lmbda * np.where(x > 0, x, alpha * (np.exp(x) - 1))  # np.where 사용 예시

y = selu(x, 1.67, 1.05) # 조절 가능.

plt.plot(x, y)
plt.grid()
plt.show()

# 한정 함수 또는 활성화 함수 라고 함