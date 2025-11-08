import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, BatchNormalization, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


a = np.array(range(1,101))

x_predict = np.array(range(96, 106))       # 101 ~ 107을 찾기!

timesteps = 6
# x.shape = n,5,1

# print(a)
# [  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
#   19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
#   37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
#   55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72
#   73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90
#   91  92  93  94  95  96  97  98  99 100]

print(a.shape)  # (100,)
print(x_predict.shape) # (10,)
print(x_predict)
# [ 96  97  98  99 100 101 102 103 104 105]


def split_xy(dataset, timesteps):
    x, y = [], []
    for i in range(len(dataset) - timesteps +1):
        x_window = dataset[i : i + timesteps -1]
        y_label = dataset[i + timesteps -1]
        x.append(x_window)
        y.append(y_label)
    return np.array(x), np.array(y)


def split_xy_predict(dataset, timesteps):
    x = []
    for i in range(len(dataset) - timesteps +1):
        subset = dataset[i : i + timesteps]
        x.append(subset)
    return np.array(x)

x, y = split_xy(a, timesteps=timesteps)
x_pred = split_xy_predict(x_predict, timesteps=5)


print("x:", x)
print("y:", y)          # 
print(x.shape, y.shape) # (95, 5) (95,)
x = x.reshape(x.shape[0], x.shape[1], 1)
# x = np.repeat(x, repeats=2, axis=2)
print(x.shape)      # (95, 5, 1)  // (95, 5, 2)



model = Sequential()
model.add(LSTM(units=100, input_shape=(5, 1), activation='relu')) #  return_sequences=True = 3차원으로 그대로 출력. 성능은 보장못함. 좋을수도 있고 나쁠수도 있음
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# model.summary()


#  3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, epochs=2000)

model.compile(loss= 'mse', optimizer='adam')


model.fit(x, y, epochs=100,
                 verbose=1,
                 validation_split=0.2,
                                  )

#  4. 평가, 예측
# results = model.evaluate(x, y)
# print('loss : ', results)


loss = model.evaluate(x, y, verbose=2)
print('loss : ', loss)



# 예측함
x_pred = x_pred.reshape(-1, 5, 1)
y_pred = model.predict(x_pred)
print('[y]의 결과 : ', y_pred) 

# loss :  0.0011848467402160168
# [y]의 결과 :  [[100.97795]
#  [101.97896]
#  [102.97981]
#  [103.98087]
#  [104.98215]
#  [105.98367]]

