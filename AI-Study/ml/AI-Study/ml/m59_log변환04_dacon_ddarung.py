import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 로그 변환 안전하게 처리
def safe_log1p(data):
    return np.log1p(np.clip(data, a_min=0, a_max=None))

def safe_expm1(data):
    return np.expm1(data)

# 1. 데이터 로드
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

train_csv = train_csv.dropna()
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

results = {}

# ------------------------
# 1. 원본 (No log)
# ------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123)
model = RandomForestRegressor(random_state=123)
model.fit(x_train, y_train)
pred = model.predict(x_test)
results['Original'] = r2_score(y_test, pred)

# ------------------------
# 2. y만 log 변환
# ------------------------
y_log = safe_log1p(y)
x_train, x_test, y_train_log, y_test_log = train_test_split(x, y_log, random_state=123)
model = RandomForestRegressor(random_state=123)
model.fit(x_train, y_train_log)
pred_log = model.predict(x_test)
pred = safe_expm1(pred_log)
_, _, y_train, y_test = train_test_split(x, y, random_state=123)  # 원래 y로 split
results['Log y only'] = r2_score(y_test, pred)

# ------------------------
# 3. x만 log 변환
# ------------------------
x_log = safe_log1p(x)
x_train, x_test, y_train, y_test = train_test_split(x_log, y, random_state=123)
model = RandomForestRegressor(random_state=123)
model.fit(x_train, y_train)
pred = model.predict(x_test)
results['Log x only'] = r2_score(y_test, pred)

# ------------------------
# 4. x, y 모두 log 변환
# ------------------------
x_log = safe_log1p(x)
y_log = safe_log1p(y)
x_train, x_test, y_train_log, y_test_log = train_test_split(x_log, y_log, random_state=123)
model = RandomForestRegressor(random_state=123)
model.fit(x_train, y_train_log)
pred_log = model.predict(x_test)
pred = safe_expm1(pred_log)
_, _, y_train, y_test = train_test_split(x, y, random_state=123)
results['Log x and y'] = r2_score(y_test, pred)

# ------------------------
# 결과 출력
# ------------------------
print("===== R² Scores (따릉이) =====")
for k, v in results.items():
    print(f"{k:15}: {v:.4f}")

# ===== R² Scores (따릉이) =====
# Original       : 0.7929
# Log y only     : 0.7735
# Log x only     : 0.7899
# Log x and y    : 0.7730