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

from sklearn.impute import SimpleImputer, KNNImputer #데이터의 거리를 가지고 보간을 하겠다 = KNN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# pip install impyute
# numpy 1.26.4 error --> 1.23.5
from impyute.imputation.cs import mice
data9 = mice(data.values,
             n=10,      # 디폴트는 5
             seed=177
             )
print(data9)
# [[ 2.          2.          2.          2.        ]
#  [ 4.          4.          4.          4.        ]
#  [ 6.          5.85185185  6.          5.55555556]
#  [ 8.          8.          8.          8.        ]
#  [10.          9.40740741 10.          8.22222222]]
