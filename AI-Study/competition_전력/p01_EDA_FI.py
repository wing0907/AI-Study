import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
import lightgbm as lgb

# ---------------------------
# matplotlib 한글 폰트 설정
# ---------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows: 맑은 고딕
plt.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

# ---------------------------
# 1. 데이터 로드 및 병합
# ---------------------------
path = 'C:/Study25/competition_전력/'
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
building_info = pd.read_csv(path + "building_info.csv")
submission = pd.read_csv(path + "sample_submission.csv")

# 용량 컬럼 수치 변환
for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    building_info[col] = building_info[col].replace('-', 0).astype(float)

# 병합
train = train.merge(building_info, on='건물번호', how='left')
test = test.merge(building_info, on='건물번호', how='left')

# ---------------------------
# 2. 시계열 피처 추가
# ---------------------------
def add_time_features(df):
    df['일시'] = pd.to_datetime(df['일시'])
    df['hour'] = df['일시'].dt.hour
    df['day'] = df['일시'].dt.day
    df['weekday'] = df['일시'].dt.weekday
    return df

train = add_time_features(train)
test = add_time_features(test)

# ---------------------------
# 3. 추가 피처 엔지니어링
# ---------------------------
def add_features(df):
    # 사인-코사인 주기 변환
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/7)
    # 냉방면적 비율
    df['냉방면적비'] = df['냉방면적(m2)'] / (df['연면적(m2)'] + 1)
    # 태양광, ESS 여부 (이진)
    df['태양광_여부'] = (df['태양광용량(kW)'] > 0).astype(int)
    df['ESS_여부'] = (df['ESS저장용량(kWh)'] > 0).astype(int)
    return df

train = add_features(train)
test = add_features(test)

# ---------------------------
# 4. EDA (간단 확인)
# ---------------------------
plt.figure(figsize=(10,5))
sns.histplot(train['전력소비량(kWh)'], bins=50, kde=True)
plt.title('전력사용량 분포')
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(x='건물유형', y='전력소비량(kWh)', data=train)
plt.xticks(rotation=45)
plt.title('건물유형별 전력사용량')
plt.show()

numeric_cols = ['기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)']
corr = train[numeric_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('기상요소와 전력소비량 상관관계')
plt.show()

# ---------------------------
# 5. 학습 데이터 준비
# ---------------------------
X = train.drop(columns=['num_date_time', '일시', '전력소비량(kWh)'])
y = train['전력소비량(kWh)']
X_test = test.drop(columns=['num_date_time', '일시'])

# 학습/검증 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# ---------------------------
# 6. SMAPE 정의
# ---------------------------
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

smape_scorer = make_scorer(smape, greater_is_better=False)

# ---------------------------
# 7. 모델 학습
# ---------------------------
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='l1', early_stopping_rounds=20, verbose=100)

# ---------------------------
# 8. 예측 및 제출
# ---------------------------
y_pred = model.predict(X_test)
submission['answer'] = y_pred
submission.to_csv("baseline_with_features.csv", index=False)
print("Baseline with features submission saved!")
