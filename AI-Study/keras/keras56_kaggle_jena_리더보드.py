import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

학생csv = 'jena_multistep_submit.csv'

path1 = 'C:\Study25\_data\kaggle\jena\\'
path2 = 'C:\Study25\_save\jena\\'

datasets = pd.read_csv(path1 + 'jena_climate_2009_2016.csv', index_col=0)

y_정답 = datasets.iloc[-144:,-1]
print(y_정답)
print(y_정답.shape)

학생꺼 = pd.read_csv(path2 + 학생csv, index_col=0)
print(학생꺼)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_정답, 학생꺼)
print('RMSE :', rmse)

# RMSE : 55.22133296993783


# RMSE : 75.79060737695532

# RMSE : 57.05567302667631


# reshape만 한거
# RMSE : 56.555880158735064

# shift한거
# RMSE : 164415.53588242698