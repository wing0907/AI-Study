import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 6, 100) # linspace는 -1 에서 6 까지 100 등분 하겠다는 뜻
print(x, len(x))

f = lambda x : x**2 - 4*x + 6
y = f(x)

plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk-', color='red')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()






