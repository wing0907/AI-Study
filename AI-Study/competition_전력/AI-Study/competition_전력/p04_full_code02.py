# ============================================================
# 규정 준수 + 누락컬럼 처리 + 7일 멀티폴드 시계열검증
# + 건물유형별 앙상블(LGBM/CatBoost/XGBoost)
# + Optuna 튜닝(하이퍼 + 앙상블 가중치) + 후처리 + 타임스탬프 제출
# ============================================================
import os
import datetime as dt
import numpy as np
import pandas as pd
from tqdm import tqdm

# ML
import optuna
from sklearn.preprocessing import LabelEncoder

# Models
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from catboost import CatBoostRegressor
import xgboost as xgb

# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = "C:/Study25/competition_전력/"
USE_GPU = False          # GPU 가능하면 True (Optuna는 n_jobs=1 권장)
N_TRIALS = 25            # Optuna trial 수 (시간 가능하면 ↑)
FOLD_DAYS = 7            # 검증 기간 길이 (7일)
N_FOLDS = 3              # 폴드 수: 3 → (8/04–8/10, 8/11–8/17, 8/18–8/24)
SEED = 42
EPS = 1e-3               # SMAPE 가중치용
SUB_SMOOTH = 0.15        # 제출 후 스무딩 비율 (기준선 혼합)

# ---------------------------
# Utils
# ---------------------------
def smape(y_true, y_pred):
    return 100 / len(y_true) * np.sum(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

def build_time_folds(df, date_col, end_date_str="2024-08-24 23:00", fold_days=7, n_folds=3):
    """끝 시점부터 거꾸로 7일 단위로 n_folds개 validation 구간 생성."""
    end_dt = pd.to_datetime(end_date_str)
    folds = []
    for i in range(n_folds):
        val_end = end_dt - pd.Timedelta(days=i*fold_days)
        val_start = val_end - pd.Timedelta(days=fold_days) + pd.Timedelta(hours=1)
        train_mask = df[date_col] < val_start
        val_mask   = (df[date_col] >= val_start) & (df[date_col] <= val_end)
        folds.append((train_mask, val_mask, (val_start, val_end)))
    folds.reverse()  # 과거→최근 순서
    return folds

# ---------------------------
# Load & Merge
# ---------------------------
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
building_info = pd.read_csv(os.path.join(DATA_PATH, "building_info.csv"))
submission_tmpl = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

# 수치 변환
for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    building_info[col] = building_info[col].replace('-', 0).astype(float)

train = train.merge(building_info, on='건물번호', how='left')
test  = test.merge(building_info, on='건물번호',  how='left')

# 일시 처리
train['일시'] = pd.to_datetime(train['일시'])
test['일시']  = pd.to_datetime(test['일시'])

# test에 없는 컬럼(일조/일사) 대비: 프레임 전체에서 보장
for miss_col in ['일조(hr)', '일사(MJ/m2)']:
    if miss_col not in test.columns:
        test[miss_col] = np.nan  # 나중에 결측 보정

# 건물유형 인코딩
le = LabelEncoder()
train['건물유형'] = le.fit_transform(train['건물유형'])
test['건물유형']  = le.transform(test['건물유형'])

# ---------------------------
# Feature Engineering
# ---------------------------
def add_time_feats(df):
    df['hour'] = df['일시'].dt.hour
    df['weekday'] = df['일시'].dt.weekday
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['weekday_sin'] = np.sin(2*np.pi*df['weekday']/7)
    df['weekday_cos'] = np.cos(2*np.pi*df['weekday']/7)
    return df

def add_building_feats(df):
    df['냉방면적비'] = df['냉방면적(m2)'] / (df['연면적(m2)'] + 1)
    df['태양광_여부'] = (df['태양광용량(kW)'] > 0).astype(int)
    df['ESS_여부']    = (df['ESS저장용량(kWh)'] > 0).astype(int)
    return df

train = add_time_feats(train)
test  = add_time_feats(test)
train = add_building_feats(train)
test  = add_building_feats(test)

# 프로파일(과거 평균) : train만으로 만들고 train/test에 조인 (규정 준수)
def build_profiles(df):
    tmp = df.copy()
    tmp['weekday'] = tmp['일시'].dt.weekday
    tmp['hour']    = tmp['일시'].dt.hour
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

# ===== Lag / Rolling (누수 방지: shift 사용) =====
# train+test 합쳐서 동일 피처 생성 (shift/rolling은 과거값만 사용 → 누수 없음)
train['is_train'] = 1
test['is_train']  = 0
all_data = pd.concat([train, test], ignore_index=True).sort_values(['건물번호','일시'])

def add_lag_rolling(df, base_cols, lags=(1,2,3,6,12,24), roll_windows=(3,24)):
    df = df.sort_values(['건물번호','일시']).copy()
    g = df.groupby('건물번호', group_keys=False)
    for col in base_cols:
        if col not in df.columns:
            df[col] = np.nan
        for lg in lags:
            df[f'{col}_lag{lg}'] = g[col].shift(lg)
        for w in roll_windows:
            # 과거 w개 평균만 사용하도록 shift(1)
            df[f'{col}_roll_mean{w}'] = g[col].apply(lambda x: x.rolling(w, min_periods=1).mean().shift(1))
    return df

all_data = add_lag_rolling(
    all_data,
    base_cols=['기온(°C)', '습도(%)', '일조(hr)', '일사(MJ/m2)'],
    lags=(1,2,3,6,12,24),
    roll_windows=(3,24)
)

# 결측치 보정: 건물별 평균 → 남으면 0
num_cols = all_data.select_dtypes(include=[np.number]).columns
for c in num_cols:
    if all_data[c].isna().any():
        all_data[c] = all_data.groupby('건물번호')[c].transform(lambda s: s.fillna(s.mean()))
        all_data[c] = all_data[c].fillna(0)

# 분리
train = all_data[all_data['is_train']==1].drop(columns=['is_train'])
test  = all_data[all_data['is_train']==0].drop(columns=['is_train'])

# 타깃/가중치
train['target_log'] = np.log1p(train['전력소비량(kWh)'])
train['w_smape'] = 1.0 / (np.abs(train['전력소비량(kWh)']) + EPS)

# 공통 피처 목록 (train·test 공통 + 학습용 컬럼 제외)
DROP_COLS = ['num_date_time','일시','전력소비량(kWh)','target_log','w_smape']
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS and c in test.columns]

# ---------------------------
# Optuna 목적함수 (유형별 × 멀티 폴드 7일 CV, LGBM 하이퍼 + 앙상블 가중치 튜닝)
# ---------------------------
def objective(trial):
    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 700, 1500),
        'learning_rate': trial.suggest_float('lgb_lr', 0.02, 0.1),
        'num_leaves': trial.suggest_int('lgb_leaves', 24, 64),
        'subsample': trial.suggest_float('lgb_subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('lgb_colsample', 0.7, 1.0),
        'min_child_samples': trial.suggest_int('lgb_min_child', 1, 20),
        'min_data_in_leaf': trial.suggest_int('lgb_min_leaf', 1, 20),
        'random_state': SEED,
        'device': 'gpu' if USE_GPU else 'cpu'
    }
    # 앙상블 가중치 (w3는 1 - w1 - w2, 음수 방지 위해 제약)
    w1 = trial.suggest_float('w1', 0.2, 0.7)
    w2 = trial.suggest_float('w2', 0.1, 0.6)
    if w1 + w2 >= 0.99:
        raise optuna.exceptions.TrialPruned()
    w3 = 1 - w1 - w2

    all_scores = []

    # 유형별 루프
    for tcode in train['건물유형'].unique():
        tr = train[train['건물유형']==tcode].copy()
        folds = build_time_folds(tr, '일시', end_date_str="2024-08-24 23:00", fold_days=FOLD_DAYS, n_folds=N_FOLDS)

        # 고정 모델 파라미터(적당히 강하지만 과하지 않게)
        cat_fixed = dict(iterations=900, learning_rate=0.05, depth=8, verbose=False,
                         task_type='GPU' if USE_GPU else 'CPU', random_state=SEED)
        xgb_fixed = dict(objective='reg:squarederror', eval_metric='mae',
                         learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8)
        if USE_GPU:
            xgb_fixed.update(tree_method='gpu_hist', predictor='gpu_predictor')

        for (tr_mask, va_mask, _rng) in folds:
            X_tr = tr.loc[tr_mask, FEATURE_COLS].fillna(0)
            y_tr = tr.loc[tr_mask, 'target_log']
            w_tr = tr.loc[tr_mask, 'w_smape']
            X_va = tr.loc[va_mask, FEATURE_COLS].fillna(0)
            y_va = tr.loc[va_mask, 'target_log']

            # LGBM
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                eval_metric='l1',
                callbacks=[early_stopping(50), log_evaluation(0)]
            )
            p_lgb = lgb_model.predict(X_va)

            # CatBoost
            cat_model = CatBoostRegressor(**cat_fixed)
            cat_model.fit(X_tr, y_tr, sample_weight=w_tr)
            p_cat = cat_model.predict(X_va)

            # XGBoost (low-level)
            dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr.values)
            dva = xgb.DMatrix(X_va, label=y_va)
            xgb_model = xgb.train(xgb_fixed, dtr, num_boost_round=900, verbose_eval=False)
            p_xgb = xgb_model.predict(dva)

            # Blend (log-space) → inverse log → clip
            p_blend_log = w1*p_lgb + w2*p_cat + w3*p_xgb
            pred = np.clip(np.expm1(p_blend_log), 0, None)
            true = np.expm1(y_va.values)
            score = smape(true, pred)
            all_scores.append(score)

    return float(np.mean(all_scores))

# ---------------------------
# Optuna 실행 (단일 스레드: GPU 충돌 방지)
# ---------------------------
print(f"[Optuna] start trials={N_TRIALS}, folds={N_FOLDS}x{FOLD_DAYS}days, USE_GPU={USE_GPU}")
with tqdm(total=N_TRIALS) as pbar:
    def _cb(study, trial):
        pbar.update(1)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[_cb], n_jobs=1)

print("[Optuna] Best params:", study.best_params)
print("[Optuna] Best value :", study.best_value)

best = study.best_params
w1, w2 = best['w1'], best['w2']
w3 = 1 - w1 - w2

# ---------------------------
# 최종 학습(전체 train) → 테스트 예측 (3모델 블렌딩)
# ---------------------------
final_ids, final_preds = [], []

for tcode in tqdm(sorted(train['건물유형'].unique()), desc="Final train/predict by type"):
    tr = train[train['건물유형']==tcode].copy()
    te = test[test['건물유형']==tcode].copy()

    X_all = tr[FEATURE_COLS].fillna(0)
    y_all = tr['target_log']
    w_all = tr['w_smape']
    X_tst = te[FEATURE_COLS].reindex(columns=FEATURE_COLS, fill_value=0)

    # LGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=best['lgb_n_estimators'],
        learning_rate=best['lgb_lr'],
        num_leaves=best['lgb_leaves'],
        subsample=best['lgb_subsample'],
        colsample_bytree=best['lgb_colsample'],
        min_child_samples=best['lgb_min_child'],
        min_data_in_leaf=best['lgb_min_leaf'],
        random_state=SEED,
        device='gpu' if USE_GPU else 'cpu'
    )
    lgb_model.fit(X_all, y_all, sample_weight=w_all, callbacks=[log_evaluation(0)])
    p_lgb = lgb_model.predict(X_tst)

    # CatBoost
    cat_model = CatBoostRegressor(
        iterations=900, learning_rate=0.05, depth=8,
        task_type='GPU' if USE_GPU else 'CPU',
        verbose=False, random_state=SEED
    )
    cat_model.fit(X_all, y_all, sample_weight=w_all)
    p_cat = cat_model.predict(X_tst)

    # XGBoost
    dtr = xgb.DMatrix(X_all, label=y_all, weight=w_all.values)
    dte = xgb.DMatrix(X_tst)
    xgb_params = dict(objective='reg:squarederror', eval_metric='mae',
                      learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8)
    if USE_GPU:
        xgb_params.update(tree_method='gpu_hist', predictor='gpu_predictor')
    xgb_model = xgb.train(xgb_params, dtr, num_boost_round=900, verbose_eval=False)
    p_xgb = xgb_model.predict(dte)

    # Blend → inverse log → clip → (선택) 기준선 스무딩
    p_log = w1*p_lgb + w2*p_cat + w3*p_xgb
    pred = np.clip(np.expm1(p_log), 0, None)

    # 기준선(요일×시간 건물 평균)과 스무딩
    base = te['prof_bld_wd_hr'].values
    pred = (1 - SUB_SMOOTH)*pred + SUB_SMOOTH*base

    final_ids.extend(te['num_date_time'].values)
    final_preds.extend(pred)

# ---------------------------
# 제출 저장 (타임스탬프)
# ---------------------------
ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
out_path = f"submission_ensemble_LGBM-CAT-XGB_{N_FOLDS}x{FOLD_DAYS}d_{ts}.csv"
sub = pd.DataFrame({'num_date_time': final_ids, 'answer': final_preds}).sort_values('num_date_time')
sub.to_csv(out_path, index=False)
print(f"[SUBMISSION] saved -> {out_path}")
