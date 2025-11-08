from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 데이터 로드
data = fetch_california_housing()
x = data.data
y = data.target

# 로그 변환 함수
def safe_log1p(data):
    return np.log1p(np.clip(data, a_min=0, a_max=None))

def safe_expm1(data):
    return np.expm1(data)

# 4가지 결과 저장용
results = {}

# -----------------------
# 1. 원본 (No log)
# -----------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123)
model = RandomForestRegressor(random_state=123)
model.fit(x_train, y_train)
pred = model.predict(x_test)
results['Original'] = r2_score(y_test, pred)

# -----------------------
# 2. y만 log 변환
# -----------------------
y_log = safe_log1p(y)
x_train, x_test, y_train_log, y_test_log = train_test_split(x, y_log, random_state=123)
model = RandomForestRegressor(random_state=123)
model.fit(x_train, y_train_log)
pred_log = model.predict(x_test)
pred = safe_expm1(pred_log)
_, _, y_train, y_test = train_test_split(x, y, random_state=123)  # 원본 y로 split
results['Log y only'] = r2_score(y_test, pred)

# -----------------------
# 3. x만 log 변환
# -----------------------
x_log = safe_log1p(x)
x_train, x_test, y_train, y_test = train_test_split(x_log, y, random_state=123)
model = RandomForestRegressor(random_state=123)
model.fit(x_train, y_train)
pred = model.predict(x_test)
results['Log x only'] = r2_score(y_test, pred)

# -----------------------
# 4. x, y 모두 log 변환
# -----------------------
x_log = safe_log1p(x)
y_log = safe_log1p(y)
x_train, x_test, y_train_log, y_test_log = train_test_split(x_log, y_log, random_state=123)
model = RandomForestRegressor(random_state=123)
model.fit(x_train, y_train_log)
pred_log = model.predict(x_test)
pred = safe_expm1(pred_log)
_, _, y_train, y_test = train_test_split(x, y, random_state=123)  # y 원래 값 기준
results['Log x and y'] = r2_score(y_test, pred)

# -----------------------
# 결과 출력
# -----------------------
print("===== R² Scores =====")
for k, v in results.items():
    print(f"{k:15}: {v:.4f}")


# ===== R² Scores =====
# Original       : 0.8105
# Log y only     : 0.8110
# Log x only     : 0.7347
# Log x and y    : 0.7366