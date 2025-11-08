# ================================================
# 2025 전력사용량 예측: 풀 파이프라인 (분리모델 + 앙상블 + Optuna + 리포트 + 후처리)
# ================================================
import os
import datetime as dt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna

# Models
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from catboost import CatBoostRegressor
import xgboost as xgb

# --------------------------------
# CONFIG
# --------------------------------
USE_GPU = True            # GPU 사용 여부 (없으면 False)
N_TRIALS = 30             # Optuna 시도 횟수
VAL_DAYS = 14             # 시간기반 검증 일수(유형별 리포트/튜닝)
DATA_PATH = 'C:/Study25/competition_전력/'

# --------------------------------
# Utils
# --------------------------------
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)))

def time_based_split(df, date_col, val_days=14):
    """마지막 val_days일 = 검증, 그 전 = 학습"""
    end_time = df[date_col].max()
    val_start = end_time - pd.Timedelta(days=val_days) + pd.Timedelta(hours=1)
    train_idx = df[date_col] < val_start
    val_idx = df[date_col] >= val_start
    return train_idx, val_idx

# --------------------------------
# Load
# --------------------------------
train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
test  = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
building_info = pd.read_csv(os.path.join(DATA_PATH, 'building_info.csv'))
submission_tmpl = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

# 수치 변환
for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    building_info[col] = building_info[col].replace('-', 0).astype(float)

# 병합
train = train.merge(building_info, on='건물번호', how='left')
test  = test.merge(building_info, on='건물번호',  how='left')

# test에 없는 일조/일사 보정
for col in ['일조(hr)', '일사(MJ/m2)']:
    if col not in test.columns:
        test[col] = 0.0

# 범주형 인코딩(일관성)
le = LabelEncoder()
train['건물유형'] = le.fit_transform(train['건물유형'])
test['건물유형']  = le.transform(test['건물유형'])

# --------------------------------
# Feature Engineering
# --------------------------------
def add_base_time_feats(df):
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
    df['ESS_여부']  = (df['ESS저장용량(kWh)'] > 0).astype(int)
    return df

train = add_base_time_feats(train)
test  = add_base_time_feats(test)

def add_weather_interactions(df):
    df['기온2'] = df['기온(°C)']**2
    df['습도2'] = df['습도(%)']**2
    df['체감지수'] = df['기온(°C)'] * (1 + 0.033*df['습도(%)']/100.0)  # 간단 proxy
    df['열지수_proxy'] = df['기온(°C)'] * df['습도(%)']
    return df

train = add_weather_interactions(train)
test  = add_weather_interactions(test)

# 프로파일 피처: 과거 평균 패턴(유형/건물별 요일x시간 기준선 등)
def build_profiles(df):
    tmp = df.copy()
    tmp['weekday'] = tmp['일시'].dt.weekday
    tmp['hour'] = tmp['일시'].dt.hour
    prof1 = tmp.groupby(['건물번호','weekday','hour'])['전력소비량(kWh)'].mean().rename('prof_bld_wd_hr').reset_index()
    prof2 = tmp.groupby(['건물번호','hour'])['전력소비량(kWh)'].mean().rename('prof_bld_hr').reset_index()
    prof3 = tmp.groupby(['건물번호'])['전력소비량(kWh)'].mean().rename('prof_bld').reset_index()
    return prof1, prof2, prof3

prof1, prof2, prof3 = build_profiles(train)

def join_profiles(df):
    out = df.merge(prof1, on=['건물번호','weekday','hour'], how='left') \
            .merge(prof2, on=['건물번호','hour'], how='left') \
            .merge(prof3, on=['건물번호'], how='left')
    for c in ['prof_bld_wd_hr','prof_bld_hr','prof_bld']:
        out[c] = out[c].fillna(out['prof_bld']).fillna(0.0)
    return out

train = join_profiles(train)
test  = join_profiles(test)

# 타깃 및 가중치(SMAPE 친화)
train['전력소비량_log'] = np.log1p(train['전력소비량(kWh)'])
EPS = 1e-3
train['w_smape'] = 1.0 / (np.abs(train['전력소비량(kWh)']) + EPS)

# 공통 피처 목록
drop_cols = ['num_date_time','일시','전력소비량(kWh)','전력소비량_log']
feature_cols = [c for c in train.columns if c not in drop_cols]

# --------------------------------
# Optuna 목적함수 (시간기반 검증 + SMAPE 가중 학습)
# --------------------------------
def objective(trial):
    smape_scores = []
    for b_type in train['건물유형'].unique():
        tr = train[train['건물유형']==b_type].copy()
        # 시간 기반 분할
        tr_idx, va_idx = time_based_split(tr, '일시', val_days=VAL_DAYS)
        X_train, X_val = tr.loc[tr_idx, feature_cols], tr.loc[va_idx, feature_cols]
        y_train, y_val = tr.loc[tr_idx, '전력소비량_log'], tr.loc[va_idx, '전력소비량_log']
        w_tr = tr.loc[tr_idx, 'w_smape']

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=...,
            learning_rate=...,
            num_leaves=...,
            subsample=...,
            colsample_bytree=...,
            min_child_samples=1,       # <=== 추가
            min_data_in_leaf=1,        # <=== 추가
            device='gpu' if USE_GPU else 'cpu'
        )
        lgb_model.fit(
            X_train, y_train,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='l1',
            callbacks=[early_stopping(50), log_evaluation(0)]
        )
        lgb_pred = lgb_model.predict(X_val)

        # CatBoost
        cat_model = CatBoostRegressor(
            iterations=trial.suggest_int('cat_iters', 600, 1500),
            learning_rate=trial.suggest_float('cat_lr', 0.01, 0.1),
            depth=trial.suggest_int('cat_depth', 6, 10),
            task_type='GPU' if USE_GPU else 'CPU',
            verbose=False
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, sample_weight=w_tr)
        cat_pred = cat_model.predict(X_val)

        # XGBoost (low-level API)
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_tr.values)
        dval   = xgb.DMatrix(X_val,   label=y_val)
        params = {
            'objective':'reg:squarederror',
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1),
            'max_depth': trial.suggest_int('xgb_depth', 6, 10),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 1.0),
            'eval_metric':'mae'
        }
        if USE_GPU:
            params.update({'tree_method':'gpu_hist', 'predictor':'gpu_predictor'})
        xgb_model = xgb.train(
            params, dtrain,
            num_boost_round=trial.suggest_int('xgb_n_estimators', 600, 1500),
            evals=[(dval,'val')], early_stopping_rounds=50, verbose_eval=False
        )
        xgb_pred = xgb_model.predict(dval)

        # 앙상블 가중치
        w1 = trial.suggest_float('w1', 0.1, 0.7)
        w2 = trial.suggest_float('w2', 0.1, 0.7)
        w3 = 1 - w1 - w2
        blended = w1*lgb_pred + w2*cat_pred + w3*xgb_pred

        smape_scores.append(smape(np.expm1(y_val), np.expm1(blended)))

    return float(np.mean(smape_scores))

# --------------------------------
# Optuna 실행 (단일 스레드: GPU 충돌 방지)
# --------------------------------
print(f"[Optuna] start: {N_TRIALS} trials, time-based CV {VAL_DAYS} days, USE_GPU={USE_GPU}")
with tqdm(total=N_TRIALS) as pbar:
    def _cb(study, trial):
        pbar.update(1)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[_cb], n_jobs=1)

print("[Optuna] Best:", study.best_params, "  value:", study.best_value)
optuna_results = study.trials_dataframe()
optuna_results.to_csv("optuna_trials_log.csv", index=False)
print("[Optuna] logged -> optuna_trials_log.csv")

best_params = study.best_params

# --------------------------------
# 유형별 검증 리포트 (시간기반) - 튜닝 파라미터로
# --------------------------------
def evaluate_by_type(train, feature_cols, best_params, device_gpu=True, val_days=14, out_csv="type_report.csv"):
    rows = []
    types = sorted(train['건물유형'].unique().tolist())
    for b_type in types:
        tr = train[train['건물유형'] == b_type].copy()
        tr_idx, va_idx = time_based_split(tr, '일시', val_days=val_days)
        X_tr, y_tr = tr.loc[tr_idx, feature_cols], tr.loc[tr_idx, '전력소비량_log']
        X_va, y_va = tr.loc[va_idx, feature_cols], tr.loc[va_idx, '전력소비량_log']
        w_tr = tr.loc[tr_idx, 'w_smape']

        # LGB
        lgb_model = lgb.LGBMRegressor(
            n_estimators=best_params['lgb_n_estimators'],
            learning_rate=best_params['lgb_lr'],
            num_leaves=best_params['lgb_leaves'],
            subsample=best_params['lgb_subsample'],
            colsample_bytree=best_params['lgb_colsample'],
            device='gpu' if device_gpu else 'cpu'
        )
        lgb_model.fit(X_tr, y_tr, sample_weight=w_tr, callbacks=[log_evaluation(0)])
        lgb_pred = lgb_model.predict(X_va)

        # Cat
        cat_model = CatBoostRegressor(
            iterations=best_params['cat_iters'],
            learning_rate=best_params['cat_lr'],
            depth=best_params['cat_depth'],
            task_type='GPU' if device_gpu else 'CPU',
            verbose=False
        )
        cat_model.fit(X_tr, y_tr, sample_weight=w_tr)
        cat_pred = cat_model.predict(X_va)

        # XGB
        dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr.values)
        dva = xgb.DMatrix(X_va)
        xgb_params = {
            'objective':'reg:squarederror',
            'learning_rate': best_params['xgb_lr'],
            'max_depth': best_params['xgb_depth'],
            'subsample': best_params['xgb_subsample'],
            'colsample_bytree': best_params['xgb_colsample'],
            'eval_metric':'mae'
        }
        if device_gpu:
            xgb_params.update({'tree_method':'gpu_hist','predictor':'gpu_predictor'})
        xgb_model = xgb.train(xgb_params, dtr, num_boost_round=best_params['xgb_n_estimators'], verbose_eval=False)
        xgb_pred = xgb_model.predict(dva)

        # Blend
        w1, w2 = best_params['w1'], best_params['w2']
        w3 = 1 - w1 - w2
        pred = w1*lgb_pred + w2*cat_pred + w3*xgb_pred
        score = smape(np.expm1(y_va), np.expm1(pred))
        rows.append({'건물유형': b_type, 'val_days': val_days, 'SMAPE': score})

    rep = pd.DataFrame(rows).sort_values('SMAPE')
    rep.to_csv(out_csv, index=False)
    print("\n===== 유형별 검증 SMAPE =====")
    print(rep)
    print(f"saved -> {out_csv}  |  mean SMAPE: {rep['SMAPE'].mean():.4f}")
    return rep

_ = evaluate_by_type(train, feature_cols, best_params, device_gpu=USE_GPU, val_days=VAL_DAYS, out_csv="type_report.csv")

# --------------------------------
# 최종 재학습 → 테스트 예측 (+후처리: 클리핑/스무딩) → 제출
# --------------------------------
final_ids, final_preds = [], []

for b_type in tqdm(train['건물유형'].unique(), desc="Final Train & Predict"):
    tr = train[train['건물유형']==b_type].copy()
    te = test[test['건물유형']==b_type].copy()

    X_all = tr[feature_cols]
    y_all = tr['전력소비량_log']
    w_all = tr['w_smape']
    X_tst = te[feature_cols]

    # LGB
    lgb_model = lgb.LGBMRegressor(
        n_estimators=best_params['lgb_n_estimators'],
        learning_rate=best_params['lgb_lr'],
        num_leaves=best_params['lgb_leaves'],
        subsample=best_params['lgb_subsample'],
        colsample_bytree=best_params['lgb_colsample'],
        device='gpu' if USE_GPU else 'cpu'
    )
    lgb_model.fit(X_all, y_all, sample_weight=w_all, callbacks=[log_evaluation(0)])
    lgb_pred = lgb_model.predict(X_tst)

    # Cat
    cat_model = CatBoostRegressor(
        iterations=best_params['cat_iters'],
        learning_rate=best_params['cat_lr'],
        depth=best_params['cat_depth'],
        task_type='GPU' if USE_GPU else 'CPU',
        verbose=False
    )
    cat_model.fit(X_all, y_all, sample_weight=w_all)
    cat_pred = cat_model.predict(X_tst)

    # XGB
    dtrain = xgb.DMatrix(X_all, label=y_all, weight=w_all.values)
    dtest  = xgb.DMatrix(X_tst)
    xgb_params = {
        'objective':'reg:squarederror',
        'learning_rate': best_params['xgb_lr'],
        'max_depth': best_params['xgb_depth'],
        'subsample': best_params['xgb_subsample'],
        'colsample_bytree': best_params['xgb_colsample'],
        'eval_metric':'mae'
    }
    if USE_GPU:
        xgb_params.update({'tree_method':'gpu_hist','predictor':'gpu_predictor'})
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=best_params['xgb_n_estimators'], verbose_eval=False)
    xgb_pred = xgb_model.predict(dtest)

    # Blend (log space)
    w1, w2 = best_params['w1'], best_params['w2']
    w3 = 1 - w1 - w2
    blended_log = w1*lgb_pred + w2*cat_pred + w3*xgb_pred

    # 역변환 + 후처리
    pred = np.expm1(blended_log)
    pred = np.clip(pred, 0, None)  # 음수 제거

    # 스무딩(기준선 섞기, 과도한 스파이크 억제)
    base = te['prof_bld_wd_hr'].values
    pred = 0.85*pred + 0.15*base

    final_ids.extend(te['num_date_time'])
    final_preds.extend(pred)

# 저장 (버전 관리)
ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
out_name = f"ensemble_optuna_submission_{ts}.csv"
sub = pd.DataFrame({'num_date_time': final_ids, 'answer': final_preds}).sort_values('num_date_time')
sub.to_csv(out_name, index=False)
print(f"[SUBMISSION] saved -> {out_name}")