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
model.add(Dense(200, input_dim=10, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=200, batch_size=32,
          verbose=1,
          validation_split=0.2)

print('=========  hist  ========')
print(hist)     # <keras.callbacks.History object at 0x0000022D52E9AC10>
print('=========  history  ========')
print(hist.history)
print('=========  loss  ========') #loss 값만 보고 싶을 경우.
print(hist.history['loss'])
print('=========  val_loss  ========')
print(hist.history['val_loss'])

# 그래프 그리기
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
matplotlib.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=(9,6))       # 9 x 6 사이즈
plt.plot(hist.history['loss'], c='red', label='loss')  # y값만 넣으면 시간순으로 그린 그림
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('당뇨 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')       #우측 상단에 label 표시
plt.grid()                          #격자 표시
plt.show()

'''
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
'''
# r2 > 0.62

# loss: 2432.34619140625
# RMSE: 49.31882171668174
# r2 스코어: 0.6223272945740537


# validation_split 사용 후
# loss: 2417.642822265625
# RMSE: 49.16952718457695
# r2 스코어: 0.6246103632402754