#  17_1 copy
import sklearn as sk
print(sk.__version__)   # 1.6.1 -> 1.1.3
 
from sklearn.datasets import load_boston   
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_boston()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

x = dataset.data
y = dataset.target

print(x)
print(x.shape)  #(506, 13)
print(y)
print(y.shape)  #(506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.25,
    shuffle=True,
    random_state=798
)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32,
          verbose=1,
          validation_split=0.2)    

# loss 와 val_loss 를 hist에 넣어준다. epoch 수의 따라 들어감. 리스트
# return에 loss 와 val_loss 가 있다.
print('=========  hist  ========')
print(hist)     # <keras.callbacks.History object at 0x0000022D52E9AC10>
print('=========  history  ========')
print(hist.history)


'''
{'loss': [114.36905670166016, 74.58294677734375, 67.28317260742188, 70.97145080566406, 70.24830627441406,
64.02484893798828, 65.46997833251953, 60.87289047241211, 61.63321304321289, 59.64462661743164],
'val_loss': [98.97097778320312, 100.52445220947266, 88.94136047363281, 85.45516967773438, 77.9679183959961,
89.77799987792969, 79.2540054321289, 78.32685089111328, 85.18476867675781, 78.31291961669922]}

'''
# dictionary 는 중괄호로 시작{} 그리고 key 와 value 의 합산이다. key는 'loss'. value는 수치(10개는 epoch의 개수 만큼 표시)
# loss의 역사를 담아서 history 라는 name이 됨. 이걸 쭉 이으면 graph 로 시각화 할 수 있음.

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
plt.title('보스턴 Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')       #우측 상단에 label 표시
plt.grid()                          #격자 표시
plt.show()





# exit()
"""
# 4. 평가, 예측
print('========================================')
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

print('loss:', loss)
print('[x]의 예측값:', results)

from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, results)
print('r2 스코어:', r2)     #0.75 이상

# validation_split 사용 전
# r2 스코어: 0.7630468715814822     

# validation_split 사용 후
# r2 스코어: 0.7542709154595346
"""