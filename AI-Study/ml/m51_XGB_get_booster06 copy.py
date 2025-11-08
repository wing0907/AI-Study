from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
# print(xgb.__version__)
from sklearn.metrics import accuracy_score, r2_score
import sklearn as sk
# print(sk.__version__) # 1.6.1
import warnings
warnings.filterwarnings('ignore')


seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y
)



es = xgb.callback.EarlyStopping(
    rounds = 50,
    metric_name = 'logloss',
    data_name = 'validation_0',
    # save_best = True,
    
)

# 2. 모델구성
model = XGBClassifier(
    n_estimators = 500,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,                 # 회귀 : rmse, mae, rmsle
    eval_metric = 'logloss',     # 다중분류 : mlogloss, merror // 이진분류 : logloss, error
                                # 2.1.1버전 이후로 fit에서 모델로 위치이동.
    callbacks = [es],
    random_state=seed
    
    
    )


model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose=0,
          )


print('acc : ', model.score(x_test, y_test))
# print(model.feature_importances_)
# [0.05029093 0.07584832 0.52197117 0.35188958]             얘네들을 순서대로 제거했을 때, 성능 나오겠죠?
                                                            # 두개를 제거했을 때, 세개를 제거했을 때, 성능 나오겠죠?
                                                            # 한번에 보게 할거야. 앞에서는 25%를 퉁 쳤죠? 그러면 중간에
                                                            # 향상 되는 걸 못 보잖아~~ 그죠옹?? 아 그렇잖아~~
                                                            # 그래서 그 구간을 찾아내는게 우리의 목적이다잉?!?!
# aaa = model.get_booster().get_score(importance_type='weight') # split 빈도수 개념
# {'f0': 121.0, 'f1': 66.0, 'f2': 11.0, 'f3': 3.0, 'f4': 28.0, 'f5': 16.0, 
#  'f6': 25.0, 'f7': 92.0, 'f8': 35.0, 'f9': 18.0, 'f10': 21.0, 'f11': 11.0, 
#  'f12': 7.0, 'f13': 32.0, 'f14': 21.0, 'f15': 13.0, 'f16': 22.0, 'f17': 8.0, 
#  'f18': 23.0, 'f19': 18.0, 'f20': 63.0, 'f21': 35.0, 'f22': 40.0, 'f23': 6.0, 
#  'f24': 18.0, 'f25': 7.0, 'f26': 20.0, 'f27': 60.0, 'f28': 22.0, 'f29': 3.0}

from sklearn.feature_selection import SelectFromModel

aaa = model.get_booster().get_score(importance_type='gain')
# {'f0': 0.09054344147443771, 'f1': 0.44619157910346985, 'f2': 0.9568471312522888, 
#  'f3': 0.22348596155643463, 'f4': 0.20814745128154755, 'f5': 0.39905595779418945, 
#  'f6': 0.027111342176795006, 'f7': 0.824626088142395, 'f8': 0.19422045350074768, 
#  'f9': 0.15491503477096558, 'f10': 0.40824103355407715, 'f11': 0.35078516602516174, 
#  'f12': 0.27346092462539673, 'f13': 0.539326548576355, 'f14': 0.14593474566936493, 
#  'f15': 0.4656308889389038, 'f16': 0.4021739661693573, 'f17': 0.1481102705001831, 
#  'f18': 0.19117841124534607, 'f19': 0.22889749705791473, 'f20': 0.5641250610351562, 
#  'f21': 0.8972993493080139, 'f22': 5.388490676879883, 'f23': 0.6300467848777771, 
#  'f24': 0.8262636661529541, 'f25': 0.1927350014448166, 'f26': 1.107054352760315, 
#  'f27': 0.9051604866981506, 'f28': 0.22356440126895905, 'f29': 0.06334438920021057}
print("Original aaa (dict):", aaa)

gain_list = list(aaa.values())      # 딕셔너리 값을 리스트로 변환
print("\ngain_list (values from dict):", gain_list)

gain_list = [aaa.get(f'f{i}', 0) for i in range(x_train.shape[1])]
gain_array = np.array(gain_list)
normalized_gain = (gain_array - gain_array.min()) / (gain_array.max() - gain_array.min())


# # --- Min-Max 정규화 적용 ---
# min_gain = np.min(gain_list)
# max_gain = np.max(gain_list)

# # 0으로 나누는 오류 방지를 위해 max_gain과 min_gain이 같은 경우 처리
# if max_gain == min_gain:
#     normalized_gain_list = np.zeros_like(gain_list, dtype=float)
# else:
#     normalized_gain_list = (gain_list - min_gain) / (max_gain - min_gain)

# print("\nnormalized_gain_list (Min-Max 정규화):", normalized_gain_list)

# 정규화된 값을 오름차순으로 정렬하여 thresholds로 사용
thresholds = np.sort(normalized_gain)
print("\nthresholds (normalized and sorted):", thresholds)


feature_names = [f'f{i}' for i in range(len(gain_list))]
df_gain = pd.DataFrame({
    'Feature': feature_names,
    'Gain': gain_list,
    'Normalized_Gain': normalized_gain
})


# 3. 정렬
df_gain = df_gain.sort_values(by='Normalized_Gain')

# 4. 기준 적용 (예: Normalized_Gain < 0.1)
drop_candidates = df_gain[df_gain['Normalized_Gain'] < 0.1]



### 정규화된 `thresholds`를 사용한 Feature Selection 및 모델 훈련

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=True) 
    mask = selection.get_support()
    reverse_mask = [not i for i in mask]
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    # Check if any features are selected to avoid errors
    if select_x_train.shape[1] == 0:
        print(f'Trech={i:.3f}: No features selected. Skipping model training.')
        continue

    select_model = XGBClassifier(
        n_estimators = 500,
        max_depth = 6,
        gamma = 0,
        min_child_weight = 0,
        subsample = 0.4,
        reg_alpha = 0,
        reg_lambda = 1,
        eval_metric = 'logloss', # XGBoost 2.1.1 버전 이후 fit에서 모델로 위치이동.
        callbacks = [es],
        random_state=seed,
    ) 
    
    select_model.fit(select_x_train, y_train,
                     eval_set = [(select_x_test, y_test)],
                     verbose = 0)
    
    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)
    print("==================================================================")
    print(f'Trech={i:.3f}, n={select_x_train.shape[1]}, ACC: {score*100:.4f}%')
    print(datasets.feature_names[reverse_mask])

    
print("🧹 제거해도 될 가능성이 높은 컬럼들:")
print(drop_candidates)
    
    
# Trech=0.002, n=30, ACC: 95.6140%
# Trech=0.004, n=29, ACC: 95.6140%
# Trech=0.005, n=28, ACC: 95.6140%
# Trech=0.008, n=27, ACC: 95.6140%
# Trech=0.008, n=26, ACC: 95.6140%
# Trech=0.009, n=25, ACC: 95.6140%
# Trech=0.011, n=24, ACC: 95.6140%
# Trech=0.011, n=23, ACC: 95.6140%
# Trech=0.011, n=22, ACC: 95.6140%
# Trech=0.012, n=21, ACC: 95.6140%
# Trech=0.013, n=20, ACC: 95.6140%
# Trech=0.013, n=19, ACC: 95.6140%
# Trech=0.013, n=18, ACC: 95.6140%
# Trech=0.016, n=17, ACC: 95.6140%
# Trech=0.020, n=16, ACC: 95.6140%
# Trech=0.023, n=15, ACC: 95.6140%
# Trech=0.023, n=14, ACC: 95.6140%
# Trech=0.023, n=13, ACC: 95.6140%
# Trech=0.026, n=12, ACC: 95.6140%
# Trech=0.027, n=11, ACC: 95.6140%
# Trech=0.031, n=10, ACC: 95.6140%
# Trech=0.032, n=9, ACC: 95.6140%
# Trech=0.036, n=8, ACC: 95.6140%
# Trech=0.047, n=7, ACC: 95.6140%
# Trech=0.047, n=6, ACC: 94.7368%
# Trech=0.051, n=5, ACC: 94.7368%
# Trech=0.052, n=4, ACC: 91.2281%
# Trech=0.055, n=3, ACC: 91.2281%
# Trech=0.063, n=2, ACC: 91.2281%
# Trech=0.308, n=1, ACC: 92.1053%







exit()
print("25%지점 : ", np.percentile(model.feature_importances_, 25)) # 25% 지점을 출력하게쒈~!
# 0.02461671084165573

percentile =  np.percentile(model.feature_importances_, 25)
print(type(percentile))     # <class 'numpy.float64'>

col_name = []
# 삭제할 컬럼(25% 이하인 놈)을 찾아내자!!!
for i, fi in enumerate(model.feature_importances_): # index 와 값(fi)
    # print(i, fi)
    if fi <= percentile:        # 값이 낮은 놈을 찾아
        col_name.append(datasets.feature_names[i])  # col_name에 집어넣어
    else:
        continue
print(col_name)         # ['sepal length (cm)']

x = pd.DataFrame(x, columns=datasets.feature_names) # 얘는 위에서 만들었어도 된다
x = x.drop(columns=col_name)                        # 고놈을 삭제해.

print(x)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    stratify=y
)

model.fit(x_train, y_train)
print('acc :', model.score(x_test, y_test))         # acc : 0.9333333333333333


# tqdm 간지나는 progress bar

# import matplotlib.pyplot as plt

# def plot_feature_importance_datasets(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     # 수평 막대 그래프, 4개의 열의 feature importance 그래프, 값 위치 센터
#     plt.yticks(np.arange(n_features), model.feature_importances_)
#     # 눈금, 숫자 레이블 표시
#     plt.xlabel("feature Importance")
#     plt.ylabel("Feature")
#     plt.ylim(-1, n_features)        # 축 범위 설정
#     plt.title(model.__class__.__name__)

# plot_feature_importance_datasets(model)
# plt.show()



