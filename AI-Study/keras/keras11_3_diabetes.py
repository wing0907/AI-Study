from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

#1. 데이터                 
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape)     #(442, 10) (442,)

# exit()
x_train, x_test, y_train, y_test = train_test_split(x, y,
                            test_size=0.25,
                            random_state=947 )      

# 2. 모델구성
model = Sequential()
model.add(Dense(200, input_dim=10))
model.add(Dense(300))
model.add(Dense(600))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

# 4. 평가, 예측
print('=====================================')
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])
print('loss:', loss)
# print('[x]의 예측값:', results)

r2 = r2_score(y_test, results)
            

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, results)    

print('RMSE:', rmse)
print('r2 스코어:', r2)  

# r2 > 0.62

# loss: 2432.34619140625
# RMSE: 49.31882171668174
# r2 스코어: 0.6223272945740537