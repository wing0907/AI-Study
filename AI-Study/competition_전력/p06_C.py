# ============================================================
# C안 Tournament Max — Safe Profiles + 병렬 스위치 + FI Pruning + 메타가중/모델 저장
# ============================================================
import os, json, random, warnings, datetime as dt

# ---------- Global threading (set BEFORE importing numpy/lightgbm/xgboost) ----------
N_THREADS = os.cpu_count() or 8
os.environ["OMP_NUM_THREADS"]      = str(N_THREADS)
os.environ["MKL_NUM_THREADS"]      = str(N_THREADS)
os.environ["NUMEXPR_NUM_THREADS"]  = str(N_THREADS)

# ---------------------------
# CONFIG (핵심 스위치)
# ---------------------------
USE_GPU               = False
OPTUNA_JOBS           = 1 if USE_GPU else min(4, N_THREADS)   # GPU면 1 권장
ENABLE_BUCKET_PARALLEL= False if USE_GPU else True            # CPU 전용 추천
MAX_BUCKET_WORKERS    = min(4, N_THREADS)

DATA_PATH   = "C:/Study25/competition_전력/"
OUT_DIR     = "./outputs_c"
MODELS_DIR  = os.path.join(OUT_DIR, "models_C")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

N_TRIALS  = 25
FOLD_DAYS = 7
N_FOLDS   = 3
SEED      = 42
EPS       = 1e-3
ALPHA_ON_RANGE  = (0.00, 0.20)
ALPHA_OFF_RANGE = (0.05, 0.35)

# --- FI Pruning (optional) ---
ENABLE_FI_PRUNING     = True          # False면 프루닝 비활성화
FI_PRUNE_BY           = "quantile"    # "quantile" 또는 "topk"
FI_PRUNE_Q            = 0.15          # 하위 15% 드롭(quantile 방식일 때)
FI_PRUNE_TOPK         = 220           # 전체에서 상위 K만 유지(topk 방식일 때)
ENABLE_QUICK_ABLATION = True          # 간단 CV로 드롭 검증(가볍게)
ABLATION_N_ESTIMATORS = 400           # Ablation 시 짧은 라운드

# ---------------------------
# Imports
# ---------------------------
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
import joblib

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from catboost import CatBoostRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")
random.seed(SEED); np.random.seed(SEED)

# ---------------------------
# Utils
# ---------------------------
def smape(y_true, y_pred):
    return 100/len(y_true)*np.sum(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)+1e-8))

def build_time_folds(df, date_col, end_date_str, fold_days=7, n_folds=3):
    end_dt=pd.to_datetime(end_date_str); folds=[]
    for i in range(n_folds):
        ve=end_dt-pd.Timedelta(days=i*fold_days)
        vs=ve-pd.Timedelta(days=fold_days)+pd.Timedelta(hours=1)
        tr=df[date_col]<vs; va=(df[date_col]>=vs)&(df[date_col]<=ve)
        folds.append((tr,va,(vs,ve)))
    folds.reverse(); return folds

def assign_time_bucket(h):
    if   0<=h<=6: return 0
    elif 7<=h<=12: return 1
    elif 13<=h<=18: return 2
    else: return 3

# ---------------------------
# Load
# ---------------------------
train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
building_info = pd.read_csv(os.path.join(DATA_PATH, "building_info.csv"))
submission_tmpl = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

for c in ['태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)']:
    building_info[c]=building_info[c].replace('-',0).astype(float)

train=train.merge(building_info,on='건물번호',how='left')
test =test.merge(building_info, on='건물번호',how='left')

train['일시']=pd.to_datetime(train['일시'])
test['일시']=pd.to_datetime(test['일시'])

# test 누락 대비
for m in ['일조(hr)','일사(MJ/m2)']:
    if m not in test.columns: test[m]=np.nan

# 라벨 인코딩
le=LabelEncoder()
train['건물유형']=le.fit_transform(train['건물유형'])
test ['건물유형']=le.transform(test['건물유형'])

# ---------------------------
# Feature Engineering
# ---------------------------
def add_time_feats(df):
    df=df.copy()
    df['hour']=df['일시'].dt.hour
    df['weekday']=df['일시'].dt.weekday
    df['month']=df['일시'].dt.month
    df['is_weekend']=(df['weekday']>=5).astype(int)
    df['hour_sin']=np.sin(2*np.pi*df['hour']/24); df['hour_cos']=np.cos(2*np.pi*df['hour']/24)
    df['weekday_sin']=np.sin(2*np.pi*df['weekday']/7); df['weekday_cos']=np.cos(2*np.pi*df['weekday']/7)
    df['is_workhour']=df['hour'].between(9,18).astype(int)
    df['time_bucket']=df['hour'].apply(assign_time_bucket)
    if ('기온(°C)' in df.columns) and ('습도(%)' in df.columns):
        T=df['기온(°C)']; RH=df['습도(%)']; df['THI']=T-(0.55-0.0055*RH)*(T-14.5)
    return df

def add_building_feats(df):
    df=df.copy()
    df['냉방면적비']=df['냉방면적(m2)']/(df['연면적(m2)']+1)
    df['태양광_여부']=(df['태양광용량(kW)']>0).astype(int)
    df['ESS_여부']   =(df['ESS저장용량(kWh)']>0).astype(int)
    return df

train=add_time_feats(train); test=add_time_feats(test)
train=add_building_feats(train); test=add_building_feats(test)

# ---- Holiday (robust) ----
def get_kr_holiday_set(start_date, end_date, data_path=None):
    start=pd.to_datetime(start_date).date(); end=pd.to_datetime(end_date).date()
    rng=pd.date_range(start,end,freq='D')
    try:
        import holidays
        years=list(range(start.year,end.year+1)); kr=holidays.KR(years=years)
        return {d.date() for d in rng if d.date() in kr}, 'holidays'
    except Exception:
        pass
    if data_path:
        p=os.path.join(data_path,'kr_holidays.csv')
        if os.path.exists(p):
            try:
                h=pd.to_datetime(pd.read_csv(p)['date']).dt.date
                return {d for d in h if start<=d<=end}, 'csv'
            except Exception:
                pass
    # fallback(주요 고정일 + 대체휴일 일요일 다음날 간단 처리)
    fixed={(1,1),(3,1),(5,5),(6,6),(8,15),(10,3),(10,9),(12,25)}
    s=set()
    for d in rng:
        if (d.month,d.day) in fixed:
            s.add(d.date())
            if d.weekday()==6 and (d+pd.Timedelta(days=1)).date()<=end:
                s.add((d+pd.Timedelta(days=1)).date())
    return s,'fallback'

def add_holiday_feats(df, holset):
    df=df.copy(); d0=pd.to_datetime(df['일시'].dt.date); hol=pd.to_datetime(sorted(list(holset)))
    df['is_holiday']=d0.isin(hol).astype(int)
    df['is_holiday_eve']=(d0+pd.Timedelta(days=1)).isin(hol).astype(int)
    df['is_holiday_next']=(d0-pd.Timedelta(days=1)).isin(hol).astype(int)
    df['is_business_day']=((df['is_weekend']==0) & (~d0.isin(hol))).astype(int)
    return df

holset, src_h = get_kr_holiday_set(train['일시'].min(), test['일시'].max(), DATA_PATH)
train=add_holiday_feats(train, holset); test=add_holiday_feats(test, holset)

# ---- CDD/HDD ----
def add_cdd_hdd(df):
    df=df.copy(); T=df['기온(°C)']
    df['CDD']=np.clip(T-18.0,0,None); df['HDD']=np.clip(18.0-T,0,None)
    g=df.groupby('건물번호',group_keys=False)
    for c in ['CDD','HDD']:
        df[f'{c}_roll24']=g[c].apply(lambda x:x.rolling(24,1).sum().shift(1))
        df[f'{c}_roll168']=g[c].apply(lambda x:x.rolling(168,1).sum().shift(1))
    df['CDD_work']=df['CDD']*df['is_workhour']
    df['HDD_business']=df['HDD']*df['is_business_day']
    df['CDD_hsin']=df['CDD']*df['hour_sin']; df['CDD_hcos']=df['CDD']*df['hour_cos']
    df['HDD_wsin']=df['HDD']*df['weekday_sin']; df['HDD_wcos']=df['HDD']*df['weekday_cos']
    df['THI_work']=df.get('THI',pd.Series(0,index=df.index))*df['is_workhour']
    return df

train=add_cdd_hdd(train); test=add_cdd_hdd(test)

# ---- Lag/Roll ----
def add_lag_roll(df, cols, lags=(1,2,3,6,12,24), rolls=(3,24)):
    df=df.sort_values(['건물번호','일시']).copy(); g=df.groupby('건물번호',group_keys=False)
    for c in cols:
        if c not in df.columns: df[c]=np.nan
        for lg in lags: df[f'{c}_lag{lg}']=g[c].shift(lg)
        for w in rolls: df[f'{c}_roll_mean{w}']=g[c].apply(lambda x:x.rolling(w,1).mean().shift(1))
    return df

for m in ['기온(°C)','습도(%)','일조(hr)','일사(MJ/m2)']:
    if m not in train.columns: train[m]=np.nan
    if m not in test.columns:  test[m]=np.nan

train['is_train']=1; test['is_train']=0
all_df=pd.concat([train,test],ignore_index=True).sort_values(['건물번호','일시'])
all_df=add_lag_roll(all_df, ['기온(°C)','습도(%)','일조(hr)','일사(MJ/m2)'])

# 결측 보정
num_cols=all_df.select_dtypes(include=[np.number]).columns
for c in num_cols:
    if all_df[c].isna().any():
        all_df[c]=all_df.groupby('건물번호')[c].transform(lambda s:s.fillna(s.mean()))
        all_df[c]=all_df[c].fillna(0)

train=all_df[all_df['is_train']==1].drop(columns=['is_train'])
test =all_df[all_df['is_train']==0].drop(columns=['is_train'])

# ---- Target/Weight ----
train['target_log']=np.log1p(train['전력소비량(kWh)'])
w_raw=1.0/(np.abs(train['전력소비량(kWh)'])+EPS); cap=np.quantile(w_raw,0.99)
train['w_smape']=np.clip(w_raw,0,cap)

# ---------------------------
# Profiles (fold-safe), Features
# ---------------------------
def build_profiles_from(df_fit):
    t=df_fit.copy()
    if 'weekday' not in t.columns: t['weekday']=t['일시'].dt.weekday
    if 'hour' not in t.columns:    t['hour']=t['일시'].dt.hour
    p1=t.groupby(['건물번호','weekday','hour'])['전력소비량(kWh)'].mean().rename('prof_bld_wd_hr').reset_index()
    p2=t.groupby(['건물번호','hour'])['전력소비량(kWh)'].mean().rename('prof_bld_hr').reset_index()
    p3=t.groupby(['건물번호'])['전력소비량(kWh)'].mean().rename('prof_bld').reset_index()
    return p1,p2,p3

def join_profiles_to(df, p1, p2, p3):
    """
    안전 병합:
    - weekday/hour 없으면 생성
    - 기존 prof_* 있으면 제거해 접미사 충돌 방지
    - 병합 실패 대비 컬럼 존재 보장 후 백오프 채움
    """
    out=df.copy()
    if 'weekday' not in out.columns: out['weekday']=out['일시'].dt.weekday
    if 'hour' not in out.columns:    out['hour']=out['일시'].dt.hour
    for c in ('prof_bld_wd_hr','prof_bld_hr','prof_bld'):
        if c in out.columns: out.drop(columns=[c], inplace=True)

    out = out.merge(p1, on=['건물번호','weekday','hour'], how='left')
    out = out.merge(p2, on=['건물번호','hour'],           how='left')
    out = out.merge(p3, on=['건물번호'],                    how='left')

    for c in ('prof_bld_wd_hr','prof_bld_hr','prof_bld'):
        if c not in out.columns: out[c]=np.nan

    # prof_bld 보강(건물 평균)
    try:
        out['prof_bld'] = out['prof_bld'].fillna(out.groupby('건물번호')['prof_bld'].transform('mean'))
    except Exception:
        pass

    for c in ('prof_bld_wd_hr','prof_bld_hr'):
        out[c]=out[c].fillna(out['prof_bld'])

    out[['prof_bld_wd_hr','prof_bld_hr','prof_bld']] = \
        out[['prof_bld_wd_hr','prof_bld_hr','prof_bld']].fillna(0.0)
    return out

DROP=['num_date_time','일시','전력소비량(kWh)','target_log','w_smape']
BASE_FEATS=[c for c in train.columns if c not in DROP]
PROF=['prof_bld_wd_hr','prof_bld_hr','prof_bld']

def make_monotone_vector(cols):
    inc = set(['연면적(m2)','CDD','CDD_roll24','CDD_roll168','HDD','HDD_roll24','HDD_roll168',
               'CDD_work','HDD_business','CDD_hsin','CDD_hcos','HDD_wsin','HDD_wcos'])
    return [1 if c in inc else 0 for c in cols]

# ---------------------------
# (선택) FI 기반 프루닝
# ---------------------------
def _compute_fi_cv_stable(df, base_feats, seed=SEED):
    feats_all = base_feats + PROF
    fi_sum = pd.Series(0.0, index=feats_all, dtype=float)
    fi_cnt = pd.Series(0,    index=feats_all, dtype=int)

    for tcode in sorted(df['건물유형'].unique()):
        dft = df[df['건물유형']==tcode].copy()
        end = str(dft['일시'].max())
        folds = build_time_folds(dft, '일시', end, FOLD_DAYS, N_FOLDS)
        for (tr_mask, va_mask, _) in folds:
            p1,p2,p3 = build_profiles_from(dft.loc[tr_mask])
            tr = join_profiles_to(dft.loc[tr_mask].copy(), p1,p2,p3)
            va = join_profiles_to(dft.loc[va_mask].copy(), p1,p2,p3)

            for b in [0,1,2,3]:
                tr_b = tr if (tr['time_bucket']==b).sum() < 200 else tr[tr['time_bucket']==b]
                if len(va[va['time_bucket']==b]) == 0:
                    continue

                X = tr_b[feats_all].fillna(0)
                y = tr_b['target_log']; w = tr_b['w_smape']

                mdl = lgb.LGBMRegressor(
                    n_estimators=600, learning_rate=0.06, num_leaves=48,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=seed, device_type='gpu' if USE_GPU else 'cpu',
                    n_jobs=N_THREADS
                )
                mdl.fit(X, y, sample_weight=w, callbacks=[log_evaluation(0)])
                fi = pd.Series(mdl.feature_importances_, index=feats_all).astype(float)
                if fi.sum() > 0: fi = fi / fi.sum()
                fi_sum = fi_sum.add(fi, fill_value=0.0)
                fi_cnt = fi_cnt.add((fi>0).astype(int), fill_value=0).astype(int)

    with np.errstate(invalid='ignore', divide='ignore'):
        fi_mean = (fi_sum / fi_cnt.replace(0, np.nan)).fillna(0.0)
    fi_presence = (fi_cnt / fi_cnt.max()).fillna(0.0)
    fi_score = (fi_mean * (0.5 + 0.5*fi_presence)).sort_values(ascending=False)
    return fi_score

def _quick_cv_smape_with_feats(df, feat_list, seed=SEED):
    scores, sizes = [], []
    for tcode in sorted(df['건물유형'].unique()):
        dft = df[df['건물유형']==tcode].copy()
        folds = build_time_folds(dft, '일시', str(dft['일시'].max()), FOLD_DAYS, N_FOLDS)
        for (tr_mask, va_mask, _) in folds:
            p1,p2,p3 = build_profiles_from(dft.loc[tr_mask])
            tr = join_profiles_to(dft.loc[tr_mask].copy(), p1,p2,p3)
            va = join_profiles_to(dft.loc[va_mask].copy(), p1,p2,p3)

            preds = np.zeros(len(va))
            for b in [0,1,2,3]:
                tr_b = tr if (tr['time_bucket']==b).sum() < 200 else tr[tr['time_bucket']==b]
                va_b = va[va['time_bucket']==b]
                if len(va_b)==0: continue

                Xtr = tr_b[feat_list + PROF].fillna(0); ytr = tr_b['target_log']; wtr = tr_b['w_smape']
                Xva = va_b[feat_list + PROF].fillna(0)

                mdl = lgb.LGBMRegressor(
                    n_estimators=ABLATION_N_ESTIMATORS, learning_rate=0.07, num_leaves=48,
                    subsample=0.8, colsample_bytree=0.8, random_state=seed,
                    device_type='gpu' if USE_GPU else 'cpu', n_jobs=N_THREADS
                )
                mdl.fit(Xtr, ytr, sample_weight=wtr, callbacks=[log_evaluation(0)])
                preds[va.index.get_indexer(va_b.index)] = np.expm1(mdl.predict(Xva))

            scores.append(smape(va['전력소비량(kWh)'].values, preds))
            sizes.append(len(va))
    return float(np.average(scores, weights=sizes))

# 프루닝 실행 (Optuna 이전)
if ENABLE_FI_PRUNING:
    print("[FI] compute stable CV FI ...")
    fi_score = _compute_fi_cv_stable(train, BASE_FEATS, seed=SEED)
    if FI_PRUNE_BY == "quantile":
        thr = fi_score.quantile(FI_PRUNE_Q)
        drop_candidates = fi_score[fi_score < thr].index.tolist()
    else:
        keep = set(fi_score.index[:FI_PRUNE_TOPK])
        drop_candidates = [c for c in fi_score.index if c not in keep]
    # PROF는 항상 유지
    drop_candidates = [c for c in drop_candidates if c not in PROF]

    if ENABLE_QUICK_ABLATION:
        base_feats0 = BASE_FEATS.copy()
        base_feats1 = [c for c in BASE_FEATS if c not in drop_candidates]
        s0 = _quick_cv_smape_with_feats(train, base_feats0, seed=SEED)
        s1 = _quick_cv_smape_with_feats(train, base_feats1, seed=SEED)
        print(f"[FI] quick ablation SMAPE: keep_all={s0:.3f} vs pruned={s1:.3f}")
        if s1 <= s0 + 1e-6:
            BASE_FEATS = base_feats1
            decided_drop = [c for c in base_feats0 if c not in base_feats1]
        else:
            decided_drop = []
    else:
        BASE_FEATS = [c for c in BASE_FEATS if c not in drop_candidates]
        decided_drop = drop_candidates

    os.makedirs(OUT_DIR, exist_ok=True)
    ts_fi = dt.datetime.now().strftime("%Y%m%d_%H%M")
    fi_report = pd.DataFrame({
        "feature": fi_score.index,
        "fi_score": fi_score.values,
        "dropped": [int(f in decided_drop) for f in fi_score.index]
    })
    fi_report.to_csv(os.path.join(OUT_DIR, f"fi_pruning_report_{ts_fi}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT_DIR, f"fi_dropped_{ts_fi}.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(decided_drop), f, ensure_ascii=False, indent=2)
    print(f"[FI] pruned {len(decided_drop)} features. Kept {len(BASE_FEATS)}")

# ---------------------------
# Optuna (메타는 최종 단계; 여기선 α/트위디/ LGB 하이퍼만)
# ---------------------------
def objective(trial):
    lgb_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 900, 1600),
        'learning_rate': trial.suggest_float('lgb_lr', 0.02, 0.12),
        'num_leaves':    trial.suggest_int('lgb_leaves', 24, 96),
        'subsample':     trial.suggest_float('lgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('lgb_colsample', 0.6, 1.0),
        'min_data_in_leaf': trial.suggest_int('lgb_min_leaf', 1, 32),
        'random_state': SEED, 'device_type':'gpu' if USE_GPU else 'cpu', 'n_jobs': N_THREADS
    }
    tw_power = trial.suggest_float('tweedie_power', 1.2, 1.6)
    alpha_on  = trial.suggest_float('alpha_on',  *ALPHA_ON_RANGE)
    alpha_off = trial.suggest_float('alpha_off', *ALPHA_OFF_RANGE)

    total_scores=[]; total_sizes=[]; MIN_BUCKET=200
    feats = BASE_FEATS + PROF
    mono  = make_monotone_vector(feats)

    for tcode in train['건물유형'].unique():
        df_t=train[train['건물유형']==tcode].copy()
        end=str(df_t['일시'].max()); folds=build_time_folds(df_t,'일시',end,FOLD_DAYS,N_FOLDS)

        for (tr_mask,va_mask,_) in folds:
            p1,p2,p3=build_profiles_from(df_t.loc[tr_mask])
            tr_f=join_profiles_to(df_t.loc[tr_mask].copy(),p1,p2,p3)
            va_f=join_profiles_to(df_t.loc[va_mask].copy(),p1,p2,p3)

            preds=np.zeros(len(va_f),dtype=float)
            for b in [0,1,2,3]:
                idx_tr_b=tr_f['time_bucket']==b; idx_va_b=va_f['time_bucket']==b
                if idx_va_b.sum()==0: continue
                use_tr = tr_f if idx_tr_b.sum()<MIN_BUCKET else tr_f.loc[idx_tr_b]

                Xtr=use_tr[feats].fillna(0); ytr_log=use_tr['target_log']; wtr=use_tr['w_smape']; ytr_org=use_tr['전력소비량(kWh)']
                Xva=va_f.loc[idx_va_b,feats].fillna(0); yva_org=va_f.loc[idx_va_b,'전력소비량(kWh)']

                lgb_m=lgb.LGBMRegressor(**lgb_params, monotone_constraints=mono)
                lgb_m.fit(Xtr,ytr_log,sample_weight=wtr,eval_set=[(Xva,np.log1p(yva_org))],
                          eval_metric='l1',callbacks=[early_stopping(100),log_evaluation(0)])
                p_lgb=np.expm1(lgb_m.predict(Xva))

                cat=CatBoostRegressor(iterations=6000,learning_rate=0.05,depth=8,
                                      task_type='GPU' if USE_GPU else 'CPU',
                                      random_state=SEED,verbose=False,loss_function='MAE',
                                      thread_count=N_THREADS)
                cat.fit(Xtr,ytr_log,sample_weight=wtr,eval_set=(Xva,np.log1p(yva_org)),
                        use_best_model=True,early_stopping_rounds=300)
                p_cat=np.expm1(cat.predict(Xva))

                dtr=xgb.DMatrix(Xtr,label=ytr_log.values,weight=wtr.values)
                dva=xgb.DMatrix(Xva,label=np.log1p(yva_org).values)
                prm=dict(objective='reg:squarederror',eval_metric='mae',learning_rate=0.05,
                         max_depth=8,subsample=0.8,colsample_bytree=0.8,seed=SEED,
                         nthread=N_THREADS)
                if USE_GPU: prm.update(tree_method='gpu_hist',predictor='gpu_predictor')
                xgb_m=xgb.train(prm,dtr,num_boost_round=12000,evals=[(dva,'valid')],
                                early_stopping_rounds=300,verbose_eval=False)
                p_xgb=np.expm1(xgb_m.predict(dva))

                tw=lgb.LGBMRegressor(objective='tweedie',tweedie_variance_power=tw_power,
                                     n_estimators=2000,learning_rate=0.05,num_leaves=64,
                                     subsample=0.8,colsample_bytree=0.8,random_state=SEED,
                                     device_type='gpu' if USE_GPU else 'cpu', n_jobs=N_THREADS,
                                     monotone_constraints=mono)
                tw.fit(Xtr,ytr_org,sample_weight=wtr,eval_set=[(Xva,yva_org)],
                       eval_metric='l1',callbacks=[early_stopping(200),log_evaluation(0)])
                p_tw=np.clip(tw.predict(Xva),0,None)

                P=(p_lgb+p_cat+p_xgb+p_tw)/4.0
                base=va_f.loc[idx_va_b,'prof_bld_wd_hr'].values
                cond=(va_f.loc[idx_va_b,'is_business_day']==0)|(va_f.loc[idx_va_b,'time_bucket'].isin([0,3]))
                alpha=np.where(cond,alpha_off,alpha_on)
                preds[va_f.index.get_indexer(va_f.loc[idx_va_b].index)] = (1-alpha)*P + alpha*base

            total_scores.append(smape(va_f['전력소비량(kWh)'].values,preds))
            total_sizes.append(len(va_f))
    return float(np.average(total_scores,weights=total_sizes))

print(f"[Optuna-C] trials={N_TRIALS}, jobs={OPTUNA_JOBS}, GPU={USE_GPU}")
sampler=optuna.samplers.TPESampler(seed=SEED)
pruner =optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=1)
with tqdm(total=N_TRIALS) as pbar:
    def _cb(study,trial): pbar.update(1)
    study=optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[_cb], n_jobs=OPTUNA_JOBS)

best=study.best_params; best_val=study.best_value
print("[Best-C]", best); print("[Val-C]", best_val)
ts=dt.datetime.now().strftime("%Y%m%d_%H%M")
with open(os.path.join(OUT_DIR, f"best_params_C_{ts}.json"), "w", encoding="utf-8") as f:
    json.dump({"best_params":best,"best_value":best_val}, f, ensure_ascii=False, indent=2)

# ---------------------------
# 최종: OOF 메타 + 테스트 예측 (버킷 병렬 옵션)
# ---------------------------
p1,p2,p3=build_profiles_from(train)
train_f=join_profiles_to(train.copy(),p1,p2,p3)
test_f =join_profiles_to(test.copy(), p1,p2,p3)
feats=BASE_FEATS+PROF
mono = make_monotone_vector(feats)

# 인코더/피처리스트 저장
joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.joblib"))
with open(os.path.join(OUT_DIR, "feature_list_C.json"), "w", encoding="utf-8") as f:
    json.dump({"features": feats, "monotone": mono}, f, ensure_ascii=False, indent=2)

# num_date_time 보정
if 'num_date_time' not in test_f.columns:
    if 'num_date_time' in submission_tmpl.columns and len(submission_tmpl)==len(test_f):
        test_f['num_date_time']=submission_tmpl['num_date_time'].values
    else:
        test_f=test_f.reset_index(drop=False).rename(columns={'index':'num_date_time'})

alpha_on, alpha_off = best['alpha_on'], best['alpha_off']
tw_power = best['tweedie_power']
MIN_BUCKET=200

final_ids=[]; final_preds=[]
meta_registry = {}

# ---- 타입별 OOF 스태킹 (순차: 안정성 우선)
for tcode in tqdm(sorted(train_f['건물유형'].unique()), desc="[C] OOF stack by type"):
    df_t=train_f[train_f['건물유형']==tcode].copy()
    end=str(df_t['일시'].max()); folds=build_time_folds(df_t,'일시',end,FOLD_DAYS,N_FOLDS)

    oof_lgb=np.zeros(len(df_t)); oof_cat=np.zeros(len(df_t))
    oof_xgb=np.zeros(len(df_t)); oof_tw =np.zeros(len(df_t))

    for (tr_mask,va_mask,_) in folds:
        p1f,p2f,p3f=build_profiles_from(df_t.loc[tr_mask])
        tr_f=join_profiles_to(df_t.loc[tr_mask].copy(),p1f,p2f,p3f)
        va_f=join_profiles_to(df_t.loc[va_mask].copy(),p1f,p2f,p3f)

        for b in [0,1,2,3]:
            idx_tr_b=tr_f['time_bucket']==b; idx_va_b=va_f['time_bucket']==b
            if idx_va_b.sum()==0: continue
            use_tr = tr_f if idx_tr_b.sum()<MIN_BUCKET else tr_f.loc[idx_tr_b]

            Xtr=use_tr[feats].fillna(0); ytr_log=use_tr['target_log']; wtr=use_tr['w_smape']; ytr_org=use_tr['전력소비량(kWh)']
            Xva=va_f.loc[idx_va_b,feats].fillna(0)

            lgb_m=lgb.LGBMRegressor(
                n_estimators=best['lgb_n_estimators'],learning_rate=best['lgb_lr'],
                num_leaves=best['lgb_leaves'],subsample=best['lgb_subsample'],
                colsample_bytree=best['lgb_colsample'],min_data_in_leaf=best['lgb_min_leaf'],
                random_state=SEED,device_type='gpu' if USE_GPU else 'cpu', n_jobs=N_THREADS,
                monotone_constraints=mono
            )
            lgb_m.fit(Xtr,ytr_log,sample_weight=wtr,callbacks=[log_evaluation(0)])
            oof_lgb[va_f.loc[idx_va_b].index]=np.expm1(lgb_m.predict(Xva))

            cat=CatBoostRegressor(iterations=900,learning_rate=0.05,depth=8,
                                  task_type='GPU' if USE_GPU else 'CPU',
                                  verbose=False,random_state=SEED,loss_function='MAE',
                                  thread_count=N_THREADS)
            cat.fit(Xtr,ytr_log,sample_weight=wtr)
            oof_cat[va_f.loc[idx_va_b].index]=np.expm1(cat.predict(Xva))

            dtr=xgb.DMatrix(Xtr,label=ytr_log.values,weight=wtr.values); dva=xgb.DMatrix(Xva)
            prm=dict(objective='reg:squarederror',eval_metric='mae',learning_rate=0.05,
                     max_depth=8,subsample=0.8,colsample_bytree=0.8,seed=SEED,
                     nthread=N_THREADS)
            if USE_GPU: prm.update(tree_method='gpu_hist',predictor='gpu_predictor')
            xgb_m=xgb.train(prm,dtr,num_boost_round=900,verbose_eval=False)
            oof_xgb[va_f.loc[idx_va_b].index]=np.expm1(xgb_m.predict(dva))

            tw=lgb.LGBMRegressor(objective='tweedie',tweedie_variance_power=tw_power,
                                 n_estimators=2000,learning_rate=0.05,num_leaves=64,
                                 subsample=0.8,colsample_bytree=0.8,random_state=SEED,
                                 device_type='gpu' if USE_GPU else 'cpu', n_jobs=N_THREADS,
                                 monotone_constraints=mono)
            tw.fit(Xtr,ytr_org,sample_weight=wtr)
            oof_tw[va_f.loc[idx_va_b].index]=np.clip(tw.predict(Xva),0,None)

    # 메타(리지) 학습 & 저장
    y_org=df_t['전력소비량(kWh)'].values
    X_meta=np.vstack([oof_lgb,oof_cat,oof_xgb,oof_tw]).T
    X_meta=np.nan_to_num(X_meta, nan=0.0, posinf=0.0, neginf=0.0)
    meta=Ridge(alpha=1.0, random_state=SEED); meta.fit(X_meta, y_org)
    w_meta=np.clip(meta.coef_,0,None); 
    w_meta=w_meta/w_meta.sum() if w_meta.sum()>0 else np.array([0.25,0.25,0.25,0.25])

    meta_registry[int(tcode)] = {"LGB": float(w_meta[0]), "CAT": float(w_meta[1]),
                                 "XGB": float(w_meta[2]), "TWD": float(w_meta[3]),
                                 "intercept": float(meta.intercept_)}

    meta_dir = os.path.join(MODELS_DIR, f"type_{tcode}")
    os.makedirs(meta_dir, exist_ok=True)
    joblib.dump(meta, os.path.join(meta_dir, "meta_ridge.joblib"))
    with open(os.path.join(meta_dir, "meta_weights.json"), "w", encoding="utf-8") as f:
        json.dump(meta_registry[int(tcode)], f, ensure_ascii=False, indent=2)

    # ---- 최종 예측 (버킷 병렬 옵션)
    te_t=test_f[test_f['건물유형']==tcode].copy()
    buckets=[0,1,2,3]

    def _fit_predict_bucket_C(b):
        tr_tb=df_t[df_t['time_bucket']==b].copy()
        te_tb=te_t[te_t['time_bucket']==b].copy()
        if len(te_tb)==0: return [], []
        if len(tr_tb)<MIN_BUCKET: tr_tb=df_t

        Xall=tr_tb[feats].fillna(0); ylog=tr_tb['target_log']; warr=tr_tb['w_smape']; yorg=tr_tb['전력소비량(kWh)']
        Xtst=te_tb[feats].fillna(0)

        lgb_m=lgb.LGBMRegressor(
            n_estimators=best['lgb_n_estimators'],learning_rate=best['lgb_lr'],
            num_leaves=best['lgb_leaves'],subsample=best['lgb_subsample'],
            colsample_bytree=best['lgb_colsample'],min_data_in_leaf=best['lgb_min_leaf'],
            random_state=SEED,device_type='gpu' if USE_GPU else 'cpu', n_jobs=N_THREADS,
            monotone_constraints=mono
        ); lgb_m.fit(Xall,ylog,sample_weight=warr,callbacks=[log_evaluation(0)])
        p_lgb=np.expm1(lgb_m.predict(Xtst))

        cat=CatBoostRegressor(iterations=900,learning_rate=0.05,depth=8,
                              task_type='GPU' if USE_GPU else 'CPU',
                              verbose=False,random_state=SEED,loss_function='MAE',
                              thread_count=N_THREADS)
        cat.fit(Xall,ylog,sample_weight=warr); p_cat=np.expm1(cat.predict(Xtst))

        dtr=xgb.DMatrix(Xall,label=ylog.values,weight=warr.values); dte=xgb.DMatrix(Xtst)
        prm=dict(objective='reg:squarederror',eval_metric='mae',learning_rate=0.05,
                 max_depth=8,subsample=0.8,colsample_bytree=0.8,seed=SEED,
                 nthread=N_THREADS)
        if USE_GPU: prm.update(tree_method='gpu_hist',predictor='gpu_predictor')
        xgb_m=xgb.train(prm,dtr,num_boost_round=900,verbose_eval=False); p_xgb=np.expm1(xgb_m.predict(dte))

        tw=lgb.LGBMRegressor(objective='tweedie',tweedie_variance_power=tw_power,
                             n_estimators=2000,learning_rate=0.05,num_leaves=64,
                             subsample=0.8,colsample_bytree=0.8,random_state=SEED,
                             device_type='gpu' if USE_GPU else 'cpu', n_jobs=N_THREADS,
                             monotone_constraints=mono)
        tw.fit(Xall,yorg,sample_weight=warr); p_tw=np.clip(tw.predict(Xtst),0,None)

        # save base models
        save_dir = os.path.join(MODELS_DIR, f"type_{tcode}", f"bucket_{b}")
        os.makedirs(save_dir, exist_ok=True)
        lgb_m.booster_.save_model(os.path.join(save_dir, "lgb.txt"))
        cat.save_model(os.path.join(save_dir, "cat.cbm"))
        xgb_m.save_model(os.path.join(save_dir, "xgb.json"))
        tw.booster_.save_model(os.path.join(save_dir, "tweedie_lgb.txt"))
        with open(os.path.join(save_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"type_code": int(tcode), "bucket": int(b), "features": feats, "monotone": mono},
                      f, ensure_ascii=False, indent=2)

        w_meta = np.array([meta_registry[int(tcode)]['LGB'],
                           meta_registry[int(tcode)]['CAT'],
                           meta_registry[int(tcode)]['XGB'],
                           meta_registry[int(tcode)]['TWD']], dtype=float)
        P = w_meta[0]*p_lgb + w_meta[1]*p_cat + w_meta[2]*p_xgb + w_meta[3]*p_tw

        base=te_tb['prof_bld_wd_hr'].values
        cond=(te_tb['is_business_day']==0)|(te_tb['time_bucket'].isin([0,3]))
        alpha=np.where(cond, alpha_off, alpha_on)
        P_sm=(1-alpha)*P + alpha*base
        return te_tb['num_date_time'].tolist(), P_sm.tolist()

    if ENABLE_BUCKET_PARALLEL:
        with ThreadPoolExecutor(max_workers=MAX_BUCKET_WORKERS) as ex:
            futures=[ex.submit(_fit_predict_bucket_C, b) for b in buckets]
            for fu in as_completed(futures):
                ids_b, preds_b = fu.result()
                final_ids.extend(ids_b); final_preds.extend(preds_b)
    else:
        for b in buckets:
            ids_b, preds_b = _fit_predict_bucket_C(b)
            final_ids.extend(ids_b); final_preds.extend(preds_b)

# 메타 레지스트리 저장
with open(os.path.join(OUT_DIR, f"meta_registry_C_{ts}.json"), "w", encoding="utf-8") as f:
    json.dump(meta_registry, f, ensure_ascii=False, indent=2)

# 제출
sub=pd.DataFrame({'num_date_time':final_ids,'answer':final_preds}).sort_values('num_date_time')
sub_path=os.path.join(OUT_DIR, f"submission_C_{N_FOLDS}x{FOLD_DAYS}_{ts}.csv")
sub.to_csv(sub_path, index=False)
print("[SUB-C] saved ->", sub_path)
print("[DONE-C]")
