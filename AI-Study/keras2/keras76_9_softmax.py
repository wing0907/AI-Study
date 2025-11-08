import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 5)  # softmax 1,2,3,4

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x)) # e의 지수승 만큼 더해서 n빵 친것
                                                                           

softmax = lambda x : np.exp(x) / np.sum(np.exp(x))
                                                  
y = softmax(x)

ratio = y
labels = y
plt.pie(ratio, labels, shadow=True, startangle=90) # softmax는 굳이 선형으로 나타내지 않아도 되서 pie 사용.
plt.grid()
plt.show()

# 한정 함수 또는 활성화 함수 라고 함