import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# 1. 데이터
datasets = load_iris()
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets['data']
y = datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names)
# print(df)
df['Target'] = y
print(df)           # [150 rows x 5 columns]

print("================= 상관관계 히트맵 짜잔 ==================")
print(df.corr())  # 복잡한 수식을 간단하게 코딩했음.

import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()      # 히트맵은 반드시 사용해서 데이터를 확인해 볼 것.


