from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

(x_train, _), (x_test, _) = mnist.load_data() # x_train만 받겠다는 것. _ 는 자리표시
print(x_train.shape, x_test.shape)      # (60000, 28, 28) (10000, 28, 28)

# 데이터 모아서 한번에 때려보자
x = np.concatenate([x_train, x_test], axis=0)
print(x.shape)      # (70000, 28, 28)

x = x.reshape(70000, 28*28) # (70000, 784)

pca = PCA(n_components=28*28) # 최대값
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
print(evr_cumsum)

# 1. 1.0 일 때 몇개?
# 2. 0.999 이상 몇개?
# 3. 0.99 이상 몇개?
# 4. 0.95 이상 몇개?

thresholds = [0.95, 0.99, 0.999, 1.0]

for threshold in thresholds:
    n_components = len(np.where(evr_cumsum >= threshold)[0]) 
    print(f"{threshold:.3f} 이상을 만족하는 주성분 개수: {n_components}개")

# 0.950 이상을 만족하는 주성분 개수: 631개
# 0.990 이상을 만족하는 주성분 개수: 454개
# 0.999 이상을 만족하는 주성분 개수: 299개
# 1.000 이상을 만족하는 주성분 개수: 72개

print('0.95 이상 : ', np.argmax(evr_cumsum>=0.950)+1)   # 0.95 이상 :  154
print('1.00 이상 : ', np.argmax(evr_cumsum>=1.000)+1)   # 1.00 이상 :  713
print('0.999 이상 : ', np.argmax(evr_cumsum>=0.999)+1)  # 0.999 이상 :  486
print('0.99 이상 : ', np.argmax(evr_cumsum>=0.990)+1)   # 0.99 이상 :  331

exit()

thresholds = [0.95, 0.99, 0.999, 1.0]

for threshold in thresholds:
    n_components = np.argmax(evr_cumsum >= threshold)
    print(f"{threshold:.3f} 이상을 만족하는 주성분 개수: {n_components}개")


# 0.950 이상을 만족하는 주성분 개수: 154개
# 0.990 이상을 만족하는 주성분 개수: 331개
# 0.999 이상을 만족하는 주성분 개수: 486개
# 1.000 이상을 만족하는 주성분 개수: 713개