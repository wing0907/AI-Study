#0
# -*- coding: utf-8 -*-
# Optuna 파라미터 저장/로드 유지, 옵튜나 1회, fold 밖에서 
# 전처리 강화 6.5x BEST VERSION

import os
import json
import random
import warnings
import datetime
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.utils import class_weight
import tensorflow as tf
from xgboost import XGBRegressor
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
# ==============================
# 0) 시드 / 경로
# ==============================
seed = 6054
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

path = 'C:/Study25/competition_전력/'
# 데이터 로드
buildinginfo = pd.read_csv(path + 'building_info.csv')
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
samplesub = pd.read_csv(path + 'sample_submission.csv')

# === 0) 옵션: building_info 병합 (있으면 병합, 없으면 넘어감)
have_bi = 'buildinginfo' in globals() or 'building_info' in globals()
if 'buildinginfo' in globals():
    bi = buildinginfo.copy()
else:
    bi = None

if bi is not None:
    # 설비 용량은 '-' → 0, 숫자로 캐스팅
    for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)', '연면적(m2)', '냉방면적(m2)']:
        if col in bi.columns:
            bi[col] = pd.to_numeric(bi[col].replace('-', np.nan), errors='coerce').fillna(0.0)
    # 설비 유무 플래그
    bi['태양광_유무'] = ((bi.get('태양광용량(kW)', 0.0).astype(float) > 0).astype(int)) if '태양광용량(kW)' in bi.columns else 0
    bi['ESS_유무']  = ((bi.get('ESS저장용량(kWh)', 0.0).astype(float) > 0).astype(int)) if 'ESS저장용량(kWh)' in bi.columns else 0

    # 필요한 컬럼만 추려 병합 (없으면 스킵)
    keep_cols = ['건물번호']
    for c in ['건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무']:
        if c in bi.columns: keep_cols.append(c)
    bi = bi[keep_cols].drop_duplicates('건물번호')

    train = train.merge(bi, on='건물번호', how='left')
    test  = test.merge(bi, on='건물번호',  how='left')

# === 1) 공통 시간 파생
def add_time_features_kor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
    df['hour']      = df['일시'].dt.hour
    df['day']       = df['일시'].dt.day
    df['month']     = df['일시'].dt.month
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['is_weekend']      = (df['dayofweek'] >= 5).astype(int)
    df['is_working_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(int)
    df['sin_hour']  = np.sin(2*np.pi*df['hour']/24)
    df['cos_hour']  = np.cos(2*np.pi*df['hour']/24)
    df['sin_month'] = np.sin(2*np.pi*df['month']/12)
    df['cos_month'] = np.cos(2*np.pi*df['month']/12)
    df['sin_dow']   = np.sin(2*np.pi*(df['dayofweek']+1)/7.0)
    df['cos_dow']   = np.cos(2*np.pi*(df['dayofweek']+1)/7.0)
    if {'기온(°C)','습도(%)'}.issubset(df.columns):
        t = df['기온(°C)']; h = df['습도(%)']
        df['DI'] = 9/5*t - 0.55*(1 - h/100.0)*(9/5*t - 26) + 32
    else:
        df['DI'] = 0.0
    return df

train = add_time_features_kor(train)
test  = add_time_features_kor(test)

# === 2) expected_solar (train 기준 → 둘 다에 머지)
if '일사(MJ/m2)' in train.columns:
    solar_proxy = (
        train.groupby(['month','hour'])['일사(MJ/m2)']
             .mean().reset_index()
             .rename(columns={'일사(MJ/m2)':'expected_solar'})
    )
    train = train.merge(solar_proxy, on=['month','hour'], how='left')
    test  = test.merge(solar_proxy,  on=['month','hour'], how='left')
else:
    train['expected_solar'] = 0.0
    test['expected_solar']  = 0.0

train['expected_solar'] = train['expected_solar'].fillna(0)
test['expected_solar']  = test['expected_solar'].fillna(0)

# === 3) 일별 온도 통계 (train/test 동일 로직)
def add_daily_temp_stats_kor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '기온(°C)' not in df.columns:
        for c in ['day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range']:
            df[c] = 0.0
        return df
    grp = df.groupby(['건물번호','month','day'])['기온(°C)']
    stats = grp.agg(day_max_temperature='max',
                      day_mean_temperature='mean',
                      day_min_temperature='min').reset_index()
    df = df.merge(stats, on=['건물번호','month','day'], how='left')
    df['day_temperature_range'] = df['day_max_temperature'] - df['day_min_temperature']
    return df

train = add_daily_temp_stats_kor(train)
test  = add_daily_temp_stats_kor(test)

# === 4) CDH / THI / WCT (train/test 동일)
def add_CDH_kor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '기온(°C)' not in df.columns:
        df['CDH'] = 0.0
        return df
    def _cdh_1d(x):
        cs = np.cumsum(x - 26)
        return np.concatenate((cs[:11], cs[11:] - cs[:-11])) if len(x) >= 12 else np.zeros_like(x, dtype=float)
    parts = []
    for bno, g in df.sort_values('일시').groupby('건물번호'):
        arr = g['기온(°C)'].to_numpy()
        cdh = _cdh_1d(arr)
        parts.append(pd.Series(cdh, index=g.index))
    df['CDH'] = pd.concat(parts).sort_index()
    return df

def add_THI_WCT_kor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {'기온(°C)','습도(%)'}.issubset(df.columns):
        t = df['기온(°C)']; h = df['습도(%)']
        df['THI'] = 9/5*t - 0.55*(1 - h/100.0)*(9/5*t - 26) + 32
    else:
        df['THI'] = 0.0
    if {'기온(°C)','풍속(m/s)'}.issubset(df.columns):
        t = df['기온(°C)']; w = df['풍속(m/s)'].clip(lower=0)
        df['WCT'] = 13.12 + 0.6125*t - 11.37*(w**0.16) + 0.3965*(w**0.16)*t
    else:
        df['WCT'] = 0.0
    return df

train = add_CDH_kor(train)
test  = add_CDH_kor(test)
train = add_THI_WCT_kor(train)
test  = add_THI_WCT_kor(test)

# === 5) 시간대 전력 통계 (train으로 계산 → 둘 다 merge)
if '전력소비량(kWh)' in train.columns:
    pm = (train
          .groupby(['건물번호','hour','dayofweek'])['전력소비량(kWh)']
          .agg(['mean','std'])
          .reset_index()
          .rename(columns={'mean':'day_hour_mean','std':'day_hour_std'}))
    train = train.merge(pm, on=['건물번호','hour','dayofweek'], how='left')
    test  = test.merge(pm,  on=['건물번호','hour','dayofweek'],  how='left')
else:
    train['day_hour_mean'] = 0.0; train['day_hour_std'] = 0.0
    test['day_hour_mean']  = 0.0; test['day_hour_std']  = 0.0

# === 6) (선택) 이상치 제거: 0 kWh 제거
if '전력소비량(kWh)' in train.columns:
    train = train.loc[train['전력소비량(kWh)'] > 0].reset_index(drop=True)

# === 7) 범주형 건물유형 인코딩 (있을 때만)
if '건물유형' in train.columns and '건물유형' in test.columns:
    both = pd.concat([train['건물유형'], test['건물유형']], axis=0).astype('category')
    cat_map = {cat: i for i, cat in enumerate(both.cat.categories)}
    train['건물유형'] = train['건물유형'].map(cat_map).fillna(-1).astype(int)
    test['건물유형']  = test['건물유형'].map(cat_map).fillna(-1).astype(int)
    
# 1) 공통 feature (train/test 둘 다 있는 컬럼만 선택)
feature_candidates = [
    '건물유형','연면적(m2)','냉방면적(m2)','태양광용량(kW)','ESS저장용량(kWh)','PCS용량(kW)','태양광_유무','ESS_유무',
    '기온(°C)','강수량(mm)','풍속(m/s)','습도(%)','일조(hr)','일사(MJ/m2)',
    'hour','day','month','dayofweek','is_weekend','is_working_hours',
    'sin_hour','cos_hour','sin_month','cos_month','sin_dow','cos_dow',
    'DI','expected_solar',
    'day_max_temperature','day_mean_temperature','day_min_temperature','day_temperature_range',
    'CDH','THI','WCT',
    'day_hour_mean','day_hour_std'
]
features = [c for c in feature_candidates if c in train.columns and c in test.columns]

# 2) target 명시
target = '전력소비량(kWh)'
if target not in train.columns:
    raise ValueError(f"train 데이터에 target 컬럼({target})이 없습니다!")

# 3) 최종 입력/타깃 데이터
X = train[features].values
y = np.log1p(train[target].values.astype(float))
X_test_raw = test[features].values
ts = train['일시']  # 내부 CV에서 정렬/참조용 가능

print(f"[확인] 사용 features 개수: {len(features)}")
print(f"[확인] target: {target}")
print(f"[확인] X shape: {X.shape}, X_test shape: {X_test_raw.shape}, y shape: {y.shape}")

# ------------------------------
# SMAPE
# ------------------------------
def smape_exp(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))

# ------------------------------
# [변경] Optuna: 내부 CV(KFold 3, shuffle=True)로 "건물당 1회" 튜닝
# ------------------------------
def create_hyperparameter():
    batchs =[16,12,8,6]
    optimizers = ['adam','rmsprop','adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu','selu',]
    node1 = [128,64,32,16]
    node2 = [128,64,32,16]
    node3 = [128,64,32,16]
    node4 = [128,64,32,16]
    node5 = [128,64,32,16, 8]
    return{
        'batch_size' : batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation': activations,
        'node1':node1,
        'node2':node2,
        'node3':node3,
        'node4':node4,
        'node5':node5,
        
    }

es = EarlyStopping(
    monitor='val_loss', # 평가 지표로 확인하겠다
    mode= 'min', # 최대값 max, 알아서 찾아줘:auto
    patience=50, # 10번까지 초과해도 넘어가겠다
    restore_best_weights= True # val_loss값이 가장 낮은 값으로 저장 해놓겠다(False시 => 최소값 이후 10번째 값으로 그냥 잡는다.)
)
rlr = ReduceLROnPlateau( # 모델이 훈련하면서 Learning Rate를 조절 할때 쓰임.
monitor='val_loss',
mode='auto',
patience= 30,
verbose=1,
factor=0.5, # Learning Rate *0.5 씩 변화를 줌(여기서는 낮아짐)
)

def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64,node3=32, node4=16, node5=8):#lr=0.001):
      #2. 모델구성
    inputs = Input(shape=(24,135),name='inputs')
    x = Conv1D(node1, kernel_size=3,activation=activation, name='conv1D')(inputs)
    x = Conv1D(node2, kernel_size=3,activation=activation, name='conv1D_1')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Bidirectional(LSTM(32))(x)
    # x = Flatten()(x)
    x = Dense(node3, activation=activation)(x)
    x= BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation)(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Dense(node5, activation=activation)(x)
    x = BatchNormalization()(x)
    outputs = Dense(1,activation='linear',name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
      # 옵티마이저 설정
    if optimizer == 'adam':
        model.compile(optimizer=Adam(learning_rate=0.01), metrics=['mae'], loss='mse')
    else:
        model.compile(optimizer=optimizer, metrics=['mae'], loss='mse')

    return model
# 기본 세팅
timesteps = 24
target_horizon = 0
stride = 1
# 슬라이딩 윈도우 함수
def split_xy_stride(X, y=None, window_size=24, stride=1):
    X_seqs = []
    y_seqs = []
    
    for i in range(0, len(X) - window_size + 1, stride):
        X_seq = X[i:i + window_size]
        
        if y is not None:  # y가 None이 아닐 때만 y를 처리
            y_seq = y[i + window_size + target_horizon - 1]
            y_seqs.append(y_seq)
        
        X_seqs.append(X_seq)
    
    if y is not None:
        return np.array(X_seqs), np.array(y_seqs)
    else:
        return np.array(X_seqs)  # X_test일 경우 y를 반환하지 않음

def process_building_kfold():

    # 1. 원핫 인코딩
    # sparse=False로 변경
    encoder = OneHotEncoder(sparse=False)
    building_encoded = encoder.fit_transform(train[["건물번호"]])
    building_encoded_test = encoder.transform(test[["건물번호"]])
    
    # 2. 원핫 인코딩된 건물번호를 기존 features에 추가
    X_full = np.concatenate([train[features].values, building_encoded], axis=1)
    y_full = np.log1p(train[target].values.astype(float))
    X_test = np.concatenate([test[features].values, building_encoded_test], axis=1)

    # 메모리 문제 해결을 위해 float32로 데이터 타입 변경
    X_full = X_full.astype(np.float32)
    y_full = y_full.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    sc = StandardScaler()
    X_full = sc.fit_transform(X_full) 
    X_test = sc.transform(X_test)
    
    X_full, y_full = split_xy_stride(X_full, y_full, window_size=timesteps, stride=stride)
    X_test = split_xy_stride(X_test, y=None, window_size=timesteps, stride=stride)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)  # 외부 KFold는 그대로 유지
    test_preds, val_smapes = [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_full), 1):
        print(f" - fold {fold}")
        X_tr, X_va = X_full[tr_idx], X_full[va_idx]
        y_tr, y_va = y_full[tr_idx], y_full[va_idx]

        X_tr_s = X_tr
        X_va_s = X_va
        X_te_s = X_test
        
        hyperparameters = create_hyperparameter()
        
        #3. 컴파일, 훈련
        keras_model = KerasRegressor(build_fn=build_model, verbose=1)

        model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, #keras를 sklearn에 랩핑한걸로 달라
                                 n_iter=50,
                                 verbose=1,
                                 refit=True
                                 )
        
        hist = model.fit(X_tr_s, y_tr,epochs= 40, batch_size= 256,verbose=1,validation_split=0.1,callbacks=[es,rlr])
        best_params = model.best_params_
        # 1. 최적의 파라미터를 로그로 기록
        print(f"Best Parameters for fold {fold}: {best_params}")
        
        # 2. 최적의 파라미터를 JSON 파일로 저장 (선택 사항)
        with open(f"best_params_fold_{fold}.json", "w") as f:
            json.dump(best_params, f, indent=4)
        
          # 스태킹을 위한 oof
        oof_tr = hist.predict(X_tr_s)
        oof_va =  hist.predict(X_va_s)
        oof_te = hist.predict(X_te_s)  
        
        # Conv1D 예측값 (None, 24, 135)을 (None, 24*135)로 변환
        # (N, T, F) -> (N, T*F) 형태로 변환
        oof_tr_flat = oof_tr.reshape(oof_tr.shape[0], -1) 
        oof_va_flat = oof_va.reshape(oof_va.shape[0], -1)  
        oof_te_flat = oof_te.reshape(oof_te.shape[0], -1)  
        
        # 3. XGBRegressor에 Conv1D 예측값을 피처로 사용하여 훈련
        xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.005, max_depth=3)
        # Conv1D 예측값을 XGB의 새로운 피처로 사용하여 훈련
        xgb_model.fit(oof_tr_flat, y_tr)
        va_pred = xgb_model.predict(oof_va_flat)  # 예측
        te_pred = xgb_model.predict(oof_te_flat)  # 예측

        fold_smape = smape_exp(y_va, va_pred)
        val_smapes.append(fold_smape)
        test_preds.append(np.expm1(te_pred))  # 역로그

    avg_test_pred = np.mean(test_preds, axis=0)
    avg_smape = float(np.mean(val_smapes)) if len(val_smapes) else np.nan
    return avg_test_pred.tolist(), avg_smape

# ==============================
# 12) 병렬 실행
# ==============================
# 모델 훈련 및 평가
final_preds, val_smapes = process_building_kfold()
# 결과를 저장할 리스트

samplesub["answer"] = final_preds
today = datetime.datetime.now().strftime("%Y%m%d")
avg_smape = float(np.mean(val_smapes))
filename = f"submission_stack_optuna_stdcols_{today}_SMAPE_{avg_smape:.4f}_{seed}.csv"
samplesub.to_csv(os.path.join(path, filename), index=False)

print(f"\n✅ 평균 SMAPE (전체 건물): {avg_smape:.4f}")
print(f"📁 저장 완료 → {filename}")