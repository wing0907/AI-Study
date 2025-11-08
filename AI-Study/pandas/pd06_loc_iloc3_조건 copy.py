import numpy as np
import matplotlib.pyplot as plt
data = np.array([-111, 2, 3, 4, 5, 6, 7, 8, 9, 1000, 11, 12, 13, 14])


# Q1, Q3, Median 계산
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
median = np.percentile(data, 50)  # 또는 np.median(data)

print("Q1 (25th percentile):", q1)
print("Q3 (75th percentile):", q3)
print("Median (50th percentile):", median)


fig = plt.figure()
fig_1 = fig.add_subplot(1,2,1) #1행 2열의 첫번째에 만들어라
fig_2 = fig.add_subplot(1,2,2)

fig_1.set_title('Original Data')
fig_1.boxplot(data)

# IQR value를 구해보아요.
iqr_value = np.percentile(data,75) - np.percentile(data,25)
print(iqr_value)  #7

# 이상치 구해보기
upper_fence = np.percentile(data,75) + (iqr_value*1.5)
lower_fence = np.percentile(data,25) - (iqr_value*1.5)

# 이상치 제거한 데이터 result
result = data[(data<upper_fence) & (data>lower_fence)]

fig_2.set_title('Remove Outlier')
fig_2.boxplot(result)

fig.tight_layout()
plt.show()


# Q1 (25th percentile): 4.25
# Q3 (75th percentile): 11.75
# Median (50th percentile): 7.5
