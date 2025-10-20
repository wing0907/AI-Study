import numpy as np
import pandas as pd

data = pd.DataFrame(
                    [[2, np.nan, 6, 8, 10],
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]]
                    )

print(data)
#      0    1    2  3     4
# 0  2.0  NaN  6.0  8  10.0
# 1  2.0  4.0  NaN  8   NaN
# 2  2.0  4.0  6.0  8  10.0
# 3  NaN  4.0  NaN  8   NaN

data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.impute import SimpleImputer, KNNImputer #데이터 사의 거리를 가지고 보간을 하겠다 = KNN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer()        # 디폴트 : BayesianRidge
data1 = imputer.fit_transform(data)
print(data1)
# [[ 2.          2.          2.          2.0000005 ]
#  [ 4.00000099  4.          4.          4.        ]
#  [ 6.          5.99999928  6.          5.9999996 ]
#  [ 8.          8.          8.          8.        ]
#  [10.          9.99999872 10.          9.99999874]]

from xgboost import XGBRegressor
xgb = XGBRegressor(
    max_depth=5,
    learning_rate=0.1,
    random_state=333,
)
imputer2 = IterativeImputer(estimator=xgb,
                            max_iter=10,
                            random_state=333,)

data2 = imputer2.fit_transform(data)
print(data2)
# [[ 2.          2.          2.          4.01184034]
#  [ 2.02664208  4.          4.          4.        ]
#  [ 6.          4.0039463   6.          4.01184034]
#  [ 8.          8.          8.          8.        ]
#  [10.          7.98026466 10.          7.98815966]]

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor