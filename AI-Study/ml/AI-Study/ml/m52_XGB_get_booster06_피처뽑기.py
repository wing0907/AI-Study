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
# print(x.shape, y.shape)  
feature_names = datasets.feature_names
# print(feature_names)


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
    # callbacks = [es],
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


score_dict = model.get_booster().get_score(importance_type='gain')
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
print(score_dict)

############### 정규화 ##############
total = sum(score_dict.values())
print(total)        # 17.476968063041568


score_list = [score_dict.get(f"f{i}", 0) / total for i in range(x.shape[1])] # 디스 이즈 안전빵!!!!
# 첫번째 f는 formatted string으로 줄여서 f-string 이라고 부른다
# 두번째 f는 그냥 문자 'f'이다
# f{i} = f + '0' => f0
print(score_list)


print(len(score_list))  # 30
thresholds = np.sort(score_list)             # 오름차순  
# print(thresholds)

###### 컬럼명 매칭 ######
# score_df = pd.DataFrame({
#     # 'feature' : [feature_names[int(f[1:])] for f in score_dict.keys()],
#     'feature' : feature_names,
#     'gain' : list(score_dict.values())
# }).sort_values(by='gain', ascending=True)
# print(score_df)

from sklearn.feature_selection import SelectFromModel


delete_columns = []
max_acc = 0
for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    mask = selection.get_support()
    # print('선택된 피처 :', mask)
    not_select_features = [feature_names[j] 
                        for j, selected in enumerate(mask) 
                        if not selected]
    

    # 삭제된 feature 이름 추출
    # mask = selection.get_support()             # 사용된 컬럼은 True
    # removed_mask = ~mask                       # 제거된 컬럼은 True
    # removed_features = feature_names[removed_mask]

    # 새 모델 학습
    select_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        gamma=0,
        min_child_weight=0,
        subsample=0.4,
        reg_alpha=0,
        reg_lambda=1,
        eval_metric='logloss',
        random_state=seed,
    )

    select_model.fit(select_x_train, y_train,
                     eval_set=[(select_x_test, y_test)],
                     verbose=0)

    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)
    
    print('--------------------------------------------------------------------------')
    print(f"Thresh={i:.3f}, n={select_x_train.shape[1]}, ACC: {score * 100:.4f}%")
    print("삭제할 컬럼 :", list(not_select_features))
    

    # 최고 점수 갱신
    if score > max_acc:
        max_acc = score
        delete_columns = not_select_features
        best_thresh = i
        best_num_features = select_x_train.shape[1]

# 최종 결과 출력
print("\n✅ 가장 성능이 높은 모델 기준")
print(f"Threshold = {best_thresh:.4f}, 선택된 feature 수 = {best_num_features}, Accuracy = {max_acc * 100:.2f}%")
print("삭제된 컬럼들:")
print(list(delete_columns))


print("=========== 끝 ===========")
print(delete_columns)
print("score : ", score)
print("정규화 gain :", best_thresh)
print("삭제할 컬럼수 :", best_num_features, "개")


# --------------------------------------------------------------------------
# Thresh=0.020, n=15, ACC: 99.1228%
# 삭제할 컬럼 : ['mean radius', 'mean area', 'mean smoothness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'texture error', 'perimeter error', 'smoothness error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst compactness', 'worst symmetry', 'worst fractal dimension']
# --------------------------------------------------------------------------


# ✅ 가장 성능이 높은 모델 기준
# Threshold = 0.0202, 선택된 feature 수 = 15, Accuracy = 99.12%
# 삭제된 컬럼들:
# ['mean radius', 'mean area', 'mean smoothness', 'mean concavity', 'mean symmetry', 'mean fractal dimension', 'texture error', 'perimeter error', 'smoothness error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst compactness', 'worst symmetry', 'worst fractal dimension']