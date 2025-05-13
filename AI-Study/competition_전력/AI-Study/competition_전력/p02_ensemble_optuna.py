import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from lightgbm import early_stopping, log_evaluation
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import datetime
import os

# ---------------------------
# 데이터 로드 & 전처리
# ---------------------------
path = 'C:/Study25/competition_전력/'
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
building_info = pd.read_csv(path + "building_info.csv").0
submission = pd.read_csv(path + "sample_submission.csv")

# 용량 수치 변환
for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    building_info[col] = building_info[col].replace('-', 0).astype(float)
train = train.merge(building_info, on='건물번호', how='left')
test = test.merge(building_info, on='건물번호', how='left')

# ---------------------------
# 누락 컬럼 처리 (test에 없는 일조/일사 추가)
# ---------------------------
for col in ['일조(hr)', '일사(MJ/m2)']:
    if col not in test.columns:
        test[col] = 0  # 예측 시 0으로 대체

# 범주형 인코딩
le = LabelEncoder()
train['건물유형'] = le.fit_transform(train['건물유형'])
test['건물유형'] = le.transform(test['건물유형'])

# 시계열/파생 피처
def add_features(df):
    df['일시'] = pd.to_datetime(df['일시'])
    df['hour'] = df['일시'].dt.hour
    df['day'] = df['일시'].dt.day
    df['weekday'] = df['일시'].dt.weekday
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['weekday_sin'] = np.sin(2*np.pi*df['weekday']/7)
    df['weekday_cos'] = np.cos(2*np.pi*df['weekday']/7)
    df['냉방면적비'] = df['냉방면적(m2)'] / (df['연면적(m2)'] + 1)
    df['태양광_여부'] = (df['태양광용량(kW)'] > 0).astype(int)
    df['ESS_여부'] = (df['ESS저장용량(kWh)'] > 0).astype(int)
    return df
train = add_features(train)
test = add_features(test)

# 로그 변환 타겟
train['전력소비량_log'] = np.log1p(train['전력소비량(kWh)'])

# ---------------------------
# 학습/예측 공통 feature list
# ---------------------------
drop_cols = ['num_date_time','일시','전력소비량(kWh)','전력소비량_log']
feature_cols = [c for c in train.columns if c not in drop_cols]

# SMAPE
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)))

# ---------------------------
# Optuna 목적함수
# ---------------------------
def objective(trial):
    smape_scores = []
    for b_type in train['건물유형'].unique():
        tr = train[train['건물유형']==b_type].copy()
        X = tr[feature_cols]
        y = tr['전력소비량_log']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=trial.suggest_int('lgb_n_estimators', 500, 1500),
            learning_rate=trial.suggest_float('lgb_lr', 0.01, 0.1),
            num_leaves=trial.suggest_int('lgb_leaves', 20, 50),
            subsample=trial.suggest_float('lgb_subsample', 0.6, 1.0),
            colsample_bytree=trial.suggest_float('lgb_colsample', 0.6, 1.0),
            device='gpu'
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='l1',
            callbacks=[early_stopping(50), log_evaluation(0)]
        )
        lgb_pred = lgb_model.predict(X_val)

        # CatBoost
        cat_model = CatBoostRegressor(
            iterations=trial.suggest_int('cat_iters', 500, 1500),
            learning_rate=trial.suggest_float('cat_lr', 0.01, 0.1),
            depth=trial.suggest_int('cat_depth', 6, 10),
            verbose=False,
            task_type='GPU'
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        cat_pred = cat_model.predict(X_val)

        # XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        params = {
            'objective':'reg:squarederror',
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1),
            'max_depth': trial.suggest_int('xgb_depth', 6, 10),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 1.0),
            'eval_metric':'mae',
            'tree_method':'gpu_hist',
            'predictor':'gpu_predictor'
        }
        xgb_model = xgb.train(
            params, dtrain,
            num_boost_round=trial.suggest_int('xgb_n_estimators', 500, 1500),
            evals=[(dval,'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        xgb_pred = xgb_model.predict(dval)

        # 앙상블 가중치
        w1 = trial.suggest_float('w1', 0.1, 0.7)
        w2 = trial.suggest_float('w2', 0.1, 0.7)
        w3 = 1 - w1 - w2
        blended = w1*lgb_pred + w2*cat_pred + w3*xgb_pred

        smape_scores.append(smape(np.expm1(y_val), np.expm1(blended)))
    return np.mean(smape_scores)

# ---------------------------
# Optuna 실행 (단일 스레드 + 진행바)
# ---------------------------
n_trials = 30
print(f"Optuna 튜닝 시작 ({n_trials} trials, 단일 스레드)...")
with tqdm(total=n_trials) as pbar:
    def tqdm_callback(study, trial):
        pbar.update(1)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback], n_jobs=1)
print("Best Params:", study.best_params)

# Optuna 결과 저장
optuna_results = study.trials_dataframe()
optuna_results.to_csv("optuna_trials_log.csv", index=False)
print("Optuna trial 로그 저장 완료: optuna_trials_log.csv")

best_params = study.best_params

# ---------------------------
# 최적 파라미터로 전체 학습 & 예측
# ---------------------------
final_preds, final_ids = [], []
for b_type in tqdm(train['건물유형'].unique(), desc="최종 모델 학습/예측"):
    tr = train[train['건물유형']==b_type].copy()
    te = test[test['건물유형']==b_type].copy()
    X = tr[feature_cols]
    y = tr['전력소비량_log']
    X_test = te[feature_cols]

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=best_params['lgb_n_estimators'],
        learning_rate=best_params['lgb_lr'],
        num_leaves=best_params['lgb_leaves'],
        subsample=best_params['lgb_subsample'],
        colsample_bytree=best_params['lgb_colsample'],
        device='gpu'
    )
    lgb_model.fit(X, y, callbacks=[log_evaluation(0)])
    lgb_pred = lgb_model.predict(X_test)

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=best_params['cat_iters'],
        learning_rate=best_params['cat_lr'],
        depth=best_params['cat_depth'],
        verbose=False,
        task_type='GPU'
    )
    cat_model.fit(X, y)
    cat_pred = cat_model.predict(X_test)

    # XGBoost
    dtrain = xgb.DMatrix(X, label=y)
    dtest = xgb.DMatrix(X_test)
    params = {
        'objective':'reg:squarederror',
        'learning_rate': best_params['xgb_lr'],
        'max_depth': best_params['xgb_depth'],
        'subsample': best_params['xgb_subsample'],
        'colsample_bytree': best_params['xgb_colsample'],
        'eval_metric':'mae',
        'tree_method':'gpu_hist',
        'predictor':'gpu_predictor'
    }
    xgb_model = xgb.train(
        params, dtrain,
        num_boost_round=best_params['xgb_n_estimators'],
        verbose_eval=False
    )
    xgb_pred = xgb_model.predict(dtest)

    # 앙상블
    w1, w2 = best_params['w1'], best_params['w2']
    w3 = 1 - w1 - w2
    blended = w1*lgb_pred + w2*cat_pred + w3*xgb_pred
    blended = np.expm1(blended)

    final_ids.extend(te['num_date_time'])
    final_preds.extend(blended)

# ---------------------------
# 제출 파일 저장 (버전 관리)
# ---------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
filename = f"ensemble_optuna_submission_{timestamp}.csv"
submission = pd.DataFrame({'num_date_time': final_ids, 'answer': final_preds})
submission = submission.sort_values('num_date_time')
submission.to_csv(filename, index=False)
print(f"최적화된 앙상블 제출 파일 저장 완료: {filename}")

# 최적화된 앙상블 제출 파일 저장 완료: ensemble_optuna_submission_20250806_1923.csv   = public 9.95983   355등
