import numpy as np
import matplotlib.pyplot as plt

aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T

print(aaa, aaa.shape)       # (13, 2)

# 함수 정의: 모든 컬럼에 대해 이상치 탐지 및 시각화
def detect_outliers_with_plot(data):
    for i in range(data.shape[1]):
        print(f"\n------ Column {i} ------")
        col = data[:, i]
        
        # 분위수 및 IQR 계산
        q1, q2, q3 = np.percentile(col, [25, 50, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 출력
        print('Q1 (25%) :', q1)
        print('Q2 (50%) :', q2)
        print('Q3 (75%) :', q3)
        print('IQR :', iqr)
        print('Lower Bound :', lower_bound)
        print('Upper Bound :', upper_bound)

        # 이상치 탐지
        outlier_idx = np.where((col < lower_bound) | (col > upper_bound))[0]
        print("이상치 인덱스:", outlier_idx)
        print("이상치 값:", col[outlier_idx])

        # Boxplot 시각화
        plt.figure()
        plt.title(f'Boxplot - Column {i}')
        plt.boxplot(col)
        plt.axhline(y=upper_bound, color='red', linestyle='--', label='Upper Bound')
        plt.axhline(y=lower_bound, color='blue', linestyle='--', label='Lower Bound')
        plt.legend()
        plt.grid()
        plt.show()

# 함수 실행
detect_outliers_with_plot(aaa)

# ------ Column 0 ------
# Q1 (25%) : 4.0
# Q2 (50%) : 7.0
# Q3 (75%) : 10.0
# IQR : 6.0
# Lower Bound : -5.0
# Upper Bound : 19.0
# 이상치 인덱스: [ 0 12]
# 이상치 값: [-10  50]

# ------ Column 1 ------
# Q1 (25%) : 200.0
# Q2 (50%) : 400.0
# Q3 (75%) : 600.0
# IQR : 400.0
# Lower Bound : -400.0
# Upper Bound : 1200.0
# 이상치 인덱스: [6]
# 이상치 값: [-70000]