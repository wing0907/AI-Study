import numpy as np
import pandas as pd
###### prerequisite : directory, name
import datetime
date = datetime.datetime.now()
# date = date.strftime({%m%d_%H%M})
path = './_save/keras30/'
mcp_name = '{epoch:04d}_{val_loss:.4f}.hdf5'
mcp_path = path + mcp_name

###### load data
from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()
input_set  = dataset.data
target_set = dataset.target

print(input_set.shape, target_set.shape) # (506, 13) (506,)

### data preprocessing
## data split
from sklearn.model_selection import train_test_split

x_tr, x_ts, y_tr, y_ts = train_test_split(input_set, target_set,
                                          train_size = 0.9, random_state = 87)

## scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
S_MM = MinMaxScaler()
S_S = StandardScaler()
S_MA = MaxAbsScaler()
S_R = RobustScaler()
r2_L = []
mse_L = []
for i in [S_MM, S_S, S_MA, S_R]:
    i.fit(x_tr)
    x_tr = i.transform(x_tr)
    x_ts = i.transform(x_ts)

    ###### modeling
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense

    model = Sequential()
    model.add(Dense(100, input_dim = x_tr.shape[1]))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(1, activation = 'relu'))


    ###### compile / fit
    model.compile(loss = 'mse', optimizer = 'adam')

    hist = model.fit(x_tr, y_tr, epochs = 50, batch_size = 64,
            validation_split = 0.1, verbose = 2 )


    ###### predicting / evaluation
    from sklearn.metrics import r2_score, mean_squared_error
    result = model.predict(x_ts)
    r2 = r2_score(result, y_ts)
    mse = mean_squared_error(result, y_ts)
    r2_L.append(r2)
    mse_L.append(mse)

print('MM : ', r2_L[0], mse_L[0])
print('S : ', r2_L[1], mse_L[1])
print('MA : ', r2_L[2], mse_L[2])
print('R : ', r2_L[3], mse_L[3])

'''
MM :  0.6833772277983894 0.29821778443785324
S :  0.7592727855067374 0.3016688596732691
MA :  0.7645881315668497 0.26209989415806667
R :  0.7644521806399711 0.27766900902549096
'''