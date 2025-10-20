import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 로그 변환 함수
def safe_log1p(data):
    return np.log1p(np.clip(data, a_min=0, a_max=None))

def safe_expm1(data):
    return np.expm1(np.clip(data, a_min=0, a_max=None))

# 평가 함수
def evaluate(true, pred):
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    return r2, rmse, mae

# -----------------------------
# 1. 데이터 로드
# -----------------------------
path = 'C:\\Study25\\_data\\kaggle\\bike\\'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)  # 사용하지 않음
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# -----------------------------
# 2. x, y 분리
# -----------------------------
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

# -----------------------------
# 3. train/test 분할 (한 번만!)
# -----------------------------
x_train_orig, x_test_orig, y_train_orig, y_test_orig = train_test_split(
    x, y, train_size=0.8, random_state=123
)

# 결과 저장용
results = {}

# -----------------------------
# 4-1. Original (no log)
# -----------------------------
model = RandomForestRegressor(random_state=123)
model.fit(x_train_orig, y_train_orig)
pred = model.predict(x_test_orig)
results['Original'] = evaluate(y_test_orig, pred)

# -----------------------------
# 4-2. Log y only
# -----------------------------
y_train_log = safe_log1p(y_train_orig)
model.fit(x_train_orig, y_train_log)
pred_log = model.predict(x_test_orig)
pred_log = np.maximum(pred_log, 0)  # 음수 방지
pred = safe_expm1(pred_log)
results['Log y only'] = evaluate(y_test_orig, pred)

# -----------------------------
# 4-3. Log x only
# -----------------------------
x_train_log = safe_log1p(x_train_orig)
x_test_log = safe_log1p(x_test_orig)
model.fit(x_train_log, y_train_orig)
pred = model.predict(x_test_log)
results['Log x only'] = evaluate(y_test_orig, pred)

# -----------------------------
# 4-4. Log x and y
# -----------------------------
y_train_log = safe_log1p(y_train_orig)
model.fit(x_train_log, y_train_log)
pred_log = model.predict(x_test_log)
pred_log = np.maximum(pred_log, 0)  # 음수 방지
pred = safe_expm1(pred_log)
results['Log x and y'] = evaluate(y_test_orig, pred)

# -----------------------------
# 5. 결과 출력
# -----------------------------
print("===== 성능 비교 (kaggle bike) =====")
print("{:<15} {:>8} {:>10} {:>10}".format("Case", "R²", "RMSE", "MAE"))
for k, (r2, rmse, mae) in results.items():
    print(f"{k:<15} {r2:8.4f} {rmse:10.4f} {mae:10.4f}")


# ===== 성능 비교 (kaggle bike) =====
# Case                  R²       RMSE        MAE
# Original          0.9998     2.5063     1.0898
# Log y only        0.9997     2.9363     1.1876
# Log x only        0.9998     2.4462     1.0567
# Log x and y       0.9997     2.8963     1.1626