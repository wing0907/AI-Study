import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# ======================
# 0. 데이터 로드
# ======================
PATH = 'C:/Study25/competition_전력/'
train = pd.read_csv(PATH + "train.csv")
test = pd.read_csv(PATH + "test.csv")
building_info = pd.read_csv(PATH + "building_info.csv")
submission = pd.read_csv(PATH + "sample_submission.csv")

# ----------------------
# 결측/형변환 처리
# ----------------------
for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    building_info[col] = building_info[col].replace('-', 0).astype(float)

train = train.merge(building_info, on='건물번호', how='left')
test = test.merge(building_info, on='건물번호', how='left')

# test에 없는 컬럼 추가 (규정 준수: 외부데이터X, 단순 결측 보완)
for col in train.columns:
    if col not in test.columns and col not in ['전력소비량(kWh)']:
        test[col] = np.nan

# ----------------------
# 피처 엔지니어링
# ----------------------
from sklearn.preprocessing import LabelEncoder

# 건물유형 인코딩
le = LabelEncoder()
train['건물유형'] = le.fit_transform(train['건물유형'])
test['건물유형'] = le.transform(test['건물유형'])

# 시간 피처
def add_time_features(df):
    df['일시'] = pd.to_datetime(df['일시'])
    df['hour'] = df['일시'].dt.hour
    df['day'] = df['일시'].dt.day
    df['weekday'] = df['일시'].dt.weekday
    return df

train = add_time_features(train)
test = add_time_features(test)

# 파생 피처
def add_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/7)
    df['냉방면적비'] = df['냉방면적(m2)'] / (df['연면적(m2)'] + 1)
    df['태양광_여부'] = (df['태양광용량(kW)'] > 0).astype(int)
    df['ESS_여부'] = (df['ESS저장용량(kWh)'] > 0).astype(int)
    return df

train = add_features(train)
test = add_features(test)

# lag 피처
def add_lag_features(df, lag_list=[1, 24]):
    cols = ['기온(°C)', '습도(%)', 'hour']
    for col in cols:
        if col in df.columns:
            for lag in lag_list:
                df[f'{col}_lag{lag}'] = df.groupby('건물번호')[col].shift(lag)
    return df

train = add_lag_features(train)
test = add_lag_features(test)

# ======================
# 1. 타깃/가중치 생성
# ======================
EPS = 1e-6
train['target_log'] = np.log1p(train['전력소비량(kWh)'])
train['w_smape'] = 1.0 / (np.abs(train['전력소비량(kWh)']) + EPS)

# ======================
# 2. FEATURE_COLS 정의
# ======================
DROP_COLS = ['num_date_time', '일시', '전력소비량(kWh)', 'target_log', 'w_smape']
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS and c in test.columns]

# ======================
# 3. SMAPE 정의
# ======================
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) /
                                    (np.abs(y_true) + np.abs(y_pred) + EPS))

smape_scorer = make_scorer(smape, greater_is_better=False)

# ======================
# 4. Optuna Objective
# ======================
def objective(trial):
    b_type = trial.suggest_int('b_type', 0, train['건물유형'].nunique()-1)

    tr = train[train['건물유형'] == b_type].copy()
    X = tr[FEATURE_COLS].fillna(0)
    y = tr['target_log']
    w = tr['w_smape']

    # 파라미터
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0)
    }

    tscv = TimeSeriesSplit(n_splits=3, test_size=24*7)
    smape_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train, w_val = w.iloc[train_idx], w.iloc[val_idx]

        model = LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            eval_metric='l1',
            callbacks=[early_stopping(50), log_evaluation(0)]
        )
        pred = np.expm1(model.predict(X_val))
        true = np.expm1(y_val)
        smape_scores.append(smape(true, pred))

    return np.mean(smape_scores)

# ======================
# 5. Optuna 실행 (간략 예시)
# ======================
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# ======================
# 6. 최종 학습 + 예측
# ======================
final_ids = []
final_preds = []

for b_type in train['건물유형'].unique():
    tr = train[train['건물유형'] == b_type].copy()
    te = test[test['건물유형'] == b_type].copy()

    X_all = tr[FEATURE_COLS].fillna(0)
    y_all = tr['target_log']
    w_all = tr['w_smape']

    X_tst = te[FEATURE_COLS].reindex(columns=FEATURE_COLS, fill_value=0)

    model = LGBMRegressor(**study.best_params)
    model.fit(X_all, y_all, sample_weight=w_all)

    pred = np.expm1(model.predict(X_tst))
    final_ids.extend(te['num_date_time'].values)
    final_preds.extend(pred)

# ======================
# 7. 제출
# ======================
submission = pd.DataFrame({'num_date_time': final_ids, 'answer': final_preds})
submission = submission.sort_values('num_date_time')
submission.to_csv(PATH + 'final_submission.csv', index=False)
print("제출 파일 저장 완료!")