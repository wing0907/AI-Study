# ============================================================
# 규정 준수 + 누락컬럼 처리 + 7일 멀티폴드 시계열검증
# + 건물유형별 앙상블(LGBM/CatBoost/XGBoost)
# + Optuna 튜닝(하이퍼 + 앙상블 가중치 + SUB_SMOOTH)
# + 휴일 피처(공휴일/전후일/비즈니스데이; robust fallback)
# + 후처리 + 시각화(Optuna/하이퍼 vs 점수/피처중요도) + 타임스탬프 제출
# ============================================================
import os
import json
import random
import warnings
import datetime as dt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Vis
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (script-safe)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'

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
DATA_PATH   = "C:/Study25/competition_전력/"
OUT_DIR     = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

USE_GPU   = False          # GPU 가능하면 True (Optuna는 n_jobs=1 권장)
N_TRIALS  = 30             # Optuna trial 수 (시간 가능하면 ↑)
FOLD_DAYS = 7              # 검증 기간 길이 (7일)
N_FOLDS   = 3              # 폴드 수
SEED      = 42
EPS       = 1e-3           # SMAPE 가중치용

# 튜닝 범위
SUB_SMOOTH_MIN, SUB_SMOOTH_MAX = 0.00, 0.35  # 스무딩 강도 범위 (CV 내에서 최적화)

warnings.filterwarnings("ignore")
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# Utils
# ---------------------------
def smape(y_true, y_pred):
    return 100 / len(y_true) * np.sum(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

def build_time_folds(df, date_col, end_date_str, fold_days=7, n_folds=3):
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
# Feature Engineering (시간/건물)
# ---------------------------
def add_time_feats(df):
    df = df.copy()
    df['hour'] = df['일시'].dt.hour
    df['weekday'] = df['일시'].dt.weekday
    df['month'] = df['일시'].dt.month
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['weekday_sin'] = np.sin(2*np.pi*df['weekday']/7)
    df['weekday_cos'] = np.cos(2*np.pi*df['weekday']/7)
    # 간단 열지수(THI) 유사 파생
    if ('기온(°C)' in df.columns) and ('습도(%)' in df.columns):
        T = df['기온(°C)']
        RH = df['습도(%)']
        df['THI'] = T - (0.55 - 0.0055*RH) * (T - 14.5)
    return df

def add_building_feats(df):
    df = df.copy()
    df['냉방면적비'] = df['냉방면적(m2)'] / (df['연면적(m2)'] + 1)
    df['태양광_여부'] = (df['태양광용량(kW)'] > 0).astype(int)
    df['ESS_여부']    = (df['ESS저장용량(kWh)'] > 0).astype(int)
    return df

train = add_time_feats(train)
test  = add_time_feats(test)
train = add_building_feats(train)
test  = add_building_feats(test)

# ---------------------------
# Holiday Features (KR 공휴일) — robust import-safe
# ---------------------------
def get_kr_holiday_set(start_date, end_date, data_path=None):
    """
    반환: (set_of_dates, source)
    source ∈ {'holidays', 'csv', 'fallback'}
    """
    start = pd.to_datetime(start_date).date()
    end   = pd.to_datetime(end_date).date()
    all_days = pd.date_range(start, end, freq='D')

    # 1) holidays 패키지 시도
    try:
        import holidays  # 미설치면 except로
        years = list(range(start.year, end.year + 1))
        kr = holidays.KR(years=years)  # 일부 버전에선 holidays.KR이 없을 수 있음
        holiday_dates = {d.date() for d in all_days if d.date() in kr}
        # 설치/동작이 됐다면 바로 반환
        return holiday_dates, 'holidays'
    except Exception:
        pass

    # 2) CSV 시도 (옵션): data_path/kr_holidays.csv, 컬럼명 'date'
    if data_path is not None:
        csv_path = os.path.join(data_path, 'kr_holidays.csv')
        if os.path.exists(csv_path):
            try:
                df_h = pd.read_csv(csv_path)
                ds = pd.to_datetime(df_h['date']).dt.date
                holiday_dates = {d for d in ds if start <= d <= end}
                return holiday_dates, 'csv'
            except Exception:
                pass

    # 3) Fallback: 고정 공휴일 + 대체휴일(일요일→월요일)
    fixed_mmdd = {(1,1),(3,1),(5,5),(6,6),(8,15),(10,3),(10,9),(12,25)}
    holiday_dates = set()
    for d in all_days:
        md = (d.month, d.day)
        if md in fixed_mmdd:
            holiday_dates.add(d.date())
            # 대체휴일(간단 규칙): 공휴일이 일요일이면 다음날 월요일 추가
            if d.weekday() == 6 and (d + pd.Timedelta(days=1)).date() <= end:
                holiday_dates.add((d + pd.Timedelta(days=1)).date())
    return holiday_dates, 'fallback'

def add_holiday_feats(df, holiday_dates):
    df = df.copy()
    d0 = pd.to_datetime(df['일시'].dt.date)
    hol = pd.to_datetime(sorted(list(holiday_dates)))

    df['is_holiday']      = d0.isin(hol).astype(int)
    df['is_holiday_eve']  = (d0 + pd.Timedelta(days=1)).isin(hol).astype(int)  # 연휴 전날
    df['is_holiday_next'] = (d0 - pd.Timedelta(days=1)).isin(hol).astype(int)  # 연휴 다음날

    # 실제 영업일(주말X & 공휴일X)
    if 'is_weekend' in df.columns:
        df['is_business_day'] = ((df['is_weekend']==0) & (df['is_holiday']==0)).astype(int)
    else:
        df['weekday'] = df['일시'].dt.weekday
        df['is_business_day'] = ((df['weekday'] < 5) & (~d0.isin(hol))).astype(int)
    return df

holidays_set, hol_src = get_kr_holiday_set(
    start_date=min(train['일시'].min(), test['일시'].min()),
    end_date=max(train['일시'].max(), test['일시'].max()),
    data_path=DATA_PATH
)
print(f"[Holiday] source = {hol_src}, count = {len(holidays_set)}")

train = add_holiday_feats(train, holidays_set)
test  = add_holiday_feats(test, holidays_set)

# ---------------------------
# Profiles (train-only로 구축 후 join)
# ---------------------------
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

# 타깃/가중치 (+ 클리핑)
train['target_log'] = np.log1p(train['전력소비량(kWh)'])
w_raw = 1.0 / (np.abs(train['전력소비량(kWh)']) + EPS)
cap = np.quantile(w_raw, 0.99)
train['w_smape'] = np.clip(w_raw, 0, cap)

# 공통 피처 목록
DROP_COLS = ['num_date_time','일시','전력소비량(kWh)','target_log','w_smape']
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS and c in test.columns]
print(f"#features = {len(FEATURE_COLS)}")

# ---------------------------
# Optuna 목적함수 (LGBM 일부 + 앙상블 가중치 + SUB_SMOOTH 튜닝)
# ---------------------------
def objective(trial):
    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 800, 1600),
        'learning_rate': trial.suggest_float('lgb_lr', 0.02, 0.12),
        'num_leaves': trial.suggest_int('lgb_leaves', 24, 96),
        'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('lgb_colsample', 0.6, 1.0),
        'min_data_in_leaf': trial.suggest_int('lgb_min_leaf', 1, 32),
        'random_state': SEED,
        'device_type': 'gpu' if USE_GPU else 'cpu',
        'n_jobs': -1
    }
    # 앙상블 가중치
    w1 = trial.suggest_float('w1', 0.2, 0.7)
    w2 = trial.suggest_float('w2', 0.1, 0.6)
    if w1 + w2 >= 0.99:
        raise optuna.exceptions.TrialPruned()
    w3 = 1 - w1 - w2

    # SUB_SMOOTH 튜닝 (기준선-스무딩 강도)
    sub_smooth = trial.suggest_float('sub_smooth', SUB_SMOOTH_MIN, SUB_SMOOTH_MAX)

    total_scores, total_sizes = [], []
    global_step = 0

    for tcode in train['건물유형'].unique():
        tr = train[train['건물유형']==tcode].copy()
        end_date = str(tr['일시'].max())
        folds = build_time_folds(tr, '일시', end_date_str=end_date, fold_days=FOLD_DAYS, n_folds=N_FOLDS)

        cat_fixed = dict(loss_function='MAE', learning_rate=0.05, depth=8,
                         task_type='GPU' if USE_GPU else 'CPU', random_state=SEED, verbose=False)
        xgb_fixed = dict(objective='reg:squarederror', eval_metric='mae',
                         learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, seed=SEED)
        if USE_GPU:
            xgb_fixed.update(tree_method='gpu_hist', predictor='gpu_predictor')

        for (tr_mask, va_mask, _rng) in folds:
            X_tr = tr.loc[tr_mask, FEATURE_COLS].fillna(0)
            y_tr = tr.loc[tr_mask, 'target_log']
            w_tr = tr.loc[tr_mask, 'w_smape']
            X_va = tr.loc[va_mask, FEATURE_COLS].fillna(0)
            y_va = tr.loc[va_mask, 'target_log']

            # LGBM (CV: early stopping)
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                eval_metric='l1',
                callbacks=[early_stopping(100), log_evaluation(0)]
            )
            p_lgb = lgb_model.predict(X_va)

            # CatBoost (CV: early stopping)
            cat_model = CatBoostRegressor(iterations=5000, **cat_fixed)
            cat_model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=(X_va, y_va),
                          use_best_model=True, early_stopping_rounds=300)
            p_cat = cat_model.predict(X_va)

            # XGBoost (CV: early stopping)
            dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr.values)
            dva = xgb.DMatrix(X_va, label=y_va)
            xgb_model = xgb.train(
                xgb_fixed, dtr, num_boost_round=12000,
                evals=[(dva, 'valid')], early_stopping_rounds=300, verbose_eval=False
            )
            p_xgb = xgb_model.predict(dva)

            # Blend (log-space)
            p_blend_log = w1*p_lgb + w2*p_cat + w3*p_xgb
            pred = np.clip(np.expm1(p_blend_log), 0, None)
            true = np.expm1(y_va.values)

            # 기준선 스무딩(튜닝 파라미터 반영)
            base = tr.loc[va_mask, 'prof_bld_wd_hr'].values
            pred = (1 - sub_smooth) * pred + sub_smooth * base

            score = smape(true, pred)
            total_scores.append(score)
            total_sizes.append(len(y_va))

            global_step += 1
            trial.report(score, step=global_step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return float(np.average(total_scores, weights=total_sizes))

# ---------------------------
# Optuna 실행
# ---------------------------
print(f"[Optuna] start trials={N_TRIALS}, folds={N_FOLDS}x{FOLD_DAYS}days, USE_GPU={USE_GPU}")
sampler = optuna.samplers.TPESampler(seed=SEED)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
with tqdm(total=N_TRIALS) as pbar:
    def _cb(study, trial):
        pbar.update(1)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[_cb], n_jobs=1)

print("[Optuna] Best params:", study.best_params)
print("[Optuna] Best value :", study.best_value)

best = study.best_params
w1, w2 = best['w1'], best['w2']
w3 = 1 - w1 - w2
SUB_SMOOTH = best['sub_smooth']  # ✅ CV로 튜닝된 스무딩 강도
print(f"[Blend Weights] w1(LGB)={w1:.3f}, w2(CAT)={w2:.3f}, w3(XGB)={w3:.3f}")
print(f"[SUB_SMOOTH] {SUB_SMOOTH:.3f}")

# 저장
ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
with open(os.path.join(OUT_DIR, f"best_params_{ts}.json"), "w", encoding="utf-8") as f:
    json.dump({"best_params": best, "best_value": study.best_value}, f, ensure_ascii=False, indent=2)

# ---------------------------
# 최종 학습(전체 train) → 테스트 예측 (3모델 블렌딩)
# + 피처중요도 집계(시각화용)
# ---------------------------
if 'num_date_time' not in test.columns:
    if 'num_date_time' in submission_tmpl.columns and len(submission_tmpl) == len(test):
        test = test.copy()
        test['num_date_time'] = submission_tmpl['num_date_time'].values
    else:
        test = test.reset_index(drop=False).rename(columns={'index': 'num_date_time'})

final_ids, final_preds = [], []

# LGBM 피처중요도 집계(유형별 학습 후 샘플 수 가중 평균)
fi_sum = pd.Series(0.0, index=FEATURE_COLS)
fi_weight_total = 0.0

for tcode in tqdm(sorted(train['건물유형'].unique()), desc="Final train/predict by type"):
    tr = train[train['건물유형']==tcode].copy()
    te = test[test['건물유형']==tcode].copy()

    X_all = tr[FEATURE_COLS].fillna(0)
    y_all = tr['target_log']
    w_all = tr['w_smape']
    X_tst = te.reindex(columns=FEATURE_COLS, fill_value=0)  # ✅ 안전 인덱싱

    # LGBM (풀학습)
    lgb_model = lgb.LGBMRegressor(
        n_estimators=best['lgb_n_estimators'],
        learning_rate=best['lgb_lr'],
        num_leaves=best['lgb_leaves'],
        subsample=best['lgb_subsample'],
        colsample_bytree=best['lgb_colsample'],
        min_data_in_leaf=best['lgb_min_leaf'],
        random_state=SEED,
        device_type='gpu' if USE_GPU else 'cpu',
        n_jobs=-1
    )
    lgb_model.fit(X_all, y_all, sample_weight=w_all, callbacks=[log_evaluation(0)])
    p_lgb = lgb_model.predict(X_tst)

    # CatBoost (풀학습)
    cat_model = CatBoostRegressor(
        iterations=900, learning_rate=0.05, depth=8,
        task_type='GPU' if USE_GPU else 'CPU',
        verbose=False, random_state=SEED, loss_function='MAE'
    )
    cat_model.fit(X_all, y_all, sample_weight=w_all)
    p_cat = cat_model.predict(X_tst)

    # XGBoost (풀학습)
    dtr = xgb.DMatrix(X_all, label=y_all, weight=w_all.values)
    dte = xgb.DMatrix(X_tst)
    xgb_params = dict(objective='reg:squarederror', eval_metric='mae',
                      learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, seed=SEED)
    if USE_GPU:
        xgb_params.update(tree_method='gpu_hist', predictor='gpu_predictor')
    xgb_model = xgb.train(xgb_params, dtr, num_boost_round=900, verbose_eval=False)
    p_xgb = xgb_model.predict(dte)

    # Blend → inverse log → clip → 기준선 스무딩(튜닝값)
    p_log = w1*p_lgb + w2*p_cat + w3*p_xgb
    pred = np.clip(np.expm1(p_log), 0, None)

    base = te['prof_bld_wd_hr'].values
    pred = (1 - SUB_SMOOTH) * pred + SUB_SMOOTH * base

    final_ids.extend(te['num_date_time'].values)
    final_preds.extend(pred)

    # ---- LGBM Feature Importance 누적 (샘플수 가중) ----
    # LightGBM은 feature_importances_ 제공
    fi = pd.Series(lgb_model.feature_importances_, index=FEATURE_COLS).astype(float)
    weight = float(len(X_all))  # 간단히 학습 샘플 수로 가중
    fi_sum = fi_sum.add(fi * weight, fill_value=0.0)
    fi_weight_total += weight

# ---------------------------
# 제출 저장 (타임스탬프)
# ---------------------------
out_sub = f"submission_ensemble_LGBM-CAT-XGB_{N_FOLDS}x{FOLD_DAYS}d_{ts}.csv"
sub = pd.DataFrame({'num_date_time': final_ids, 'answer': final_preds}).sort_values('num_date_time')
sub.to_csv(out_sub, index=False)
print(f"[SUBMISSION] saved -> {out_sub}")

# ============================================================
# 시각화: Optuna 히스토리 / 하이퍼 vs 점수 / LGBM FI Top20
# ============================================================
# Optuna trials dataframe
try:
    df_trials = study.trials_dataframe()
    df_trials.to_csv(os.path.join(OUT_DIR, f"optuna_trials_{ts}.csv"), index=False)
except Exception:
    df_trials = None

# 1) Optimization history (trial vs value)
plt.figure(figsize=(8,5))
values = [t.value for t in study.trials if t.value is not None]
plt.plot(range(1, len(values)+1), values, marker='o')
plt.title("Optuna Optimization History (SMAPE ↓)")
plt.xlabel("Trial")
plt.ylabel("Validation SMAPE")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"viz_optuna_history_{ts}.png"))

# 2) Hyperparameters vs Score (scatter): w1, w2, sub_smooth
def _scatter_param(param_name, fname):
    xs, ys = [], []
    for t in study.trials:
        if t.value is None: 
            continue
        if param_name in t.params:
            xs.append(t.params[param_name])
            ys.append(t.value)
    if len(xs) > 0:
        plt.figure(figsize=(6,5))
        plt.scatter(xs, ys, s=22)
        plt.title(f"{param_name} vs SMAPE (lower is better)")
        plt.xlabel(param_name)
        plt.ylabel("Validation SMAPE")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{fname}_{ts}.png"))

_scatter_param('w1', 'viz_w1_vs_smape')
_scatter_param('w2', 'viz_w2_vs_smape')
_scatter_param('sub_smooth', 'viz_subsmooth_vs_smape')

# 3) LGBM Feature Importance (weighted avg) Top20
if fi_weight_total > 0:
    fi_avg = (fi_sum / fi_weight_total)
    fi_top = fi_avg.sort_values(ascending=False).head(20)
    plt.figure(figsize=(9,7))
    fi_top[::-1].plot(kind='barh')  # reverse for top-down
    plt.title("LGBM Feature Importance (Weighted Avg across Types) - Top 20")
    plt.xlabel("Importance (avg)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"viz_lgbm_fi_top20_{ts}.png"))

print(f"[VIZ] saved to {OUT_DIR}")
print("[DONE]")
