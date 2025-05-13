from sklearn.datasets import load_iris, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split

seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)  # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=seed,
    # stratify=y
)


# 2. 모델구성
model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRegressor(random_state=seed)

models = [model1, model2, model3, model4]
for model in models:
    model.fit(x_train, y_train)
    print("================", model.__class__.__name__, "================")
    print('r2 : ', model.score(x_test, y_test))
    print(model.feature_importances_)

# ================ DecisionTreeRegressor ================
# r2 :  0.15795709914946876
# [0.09559417 0.01904038 0.23114463 0.0534315  0.03604905 0.05879742
#  0.04902482 0.01682605 0.36525519 0.07483678]
# ================ RandomForestRegressor ================
# r2 :  0.5265549614751442
# [0.05770917 0.01047587 0.28528549 0.09846103 0.04390962 0.05190847
#  0.05713042 0.02626033 0.28720491 0.08165469]
# ================ GradientBoostingRegressor ================
# r2 :  0.5585049159804119
# [0.04935014 0.01077655 0.30278452 0.11174122 0.02686628 0.05718503
#  0.04058792 0.01773638 0.33840513 0.04456684]
# ================ XGBRegressor ================
# r2 :  0.39065385219018145
# [0.04159961 0.07224615 0.17835377 0.06647415 0.04094251 0.04973729
#  0.03822911 0.10475955 0.3368922  0.07076568]