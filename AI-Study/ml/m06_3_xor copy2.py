from sklearn.linear_model import Perceptron
import numpy as np

# XOR 원본
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

# OR perceptron
or_model = Perceptron()
or_y = np.array([0,1,1,1])
or_model.fit(x, or_y)
or_out = or_model.predict(x).reshape(-1,1)

# AND perceptron
and_model = Perceptron()
and_y = np.array([0,0,0,1])
and_model.fit(x, and_y)
and_out = and_model.predict(x).reshape(-1,1)

# XOR = OR and NOT(AND)
xor_input = np.concatenate([or_out, 1 - and_out], axis=1)

# 최종 XOR perceptron
xor_model = Perceptron()
xor_model.fit(xor_input, y)
y_pred = xor_model.predict(xor_input)

from sklearn.metrics import accuracy_score
print("acc:", accuracy_score(y, y_pred))  # 1.0


# acc: 1.0