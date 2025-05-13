import numpy as np
import matplotlib.pyplot as plt

aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T

print(aaa)
from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=0.1) # 디폴트 0.1 -> 약10%

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)
# [ 1  1  1  1  1  1 -1  1  1  1  1  1 -1]   7번째 조합과 마지막 번째 조합이 이상치이다.