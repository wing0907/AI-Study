from sklearn.datasets import load_iris, load_diabetes, fetch_california_housing, load_wine, load_digits
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


seed = 123
random.seed(seed)
np.random.seed(seed)

# 1. 데이터
data1 = fetch_california_housing()
data2 = load_diabetes()

datasets = [data1, data2]
dataset_name = ['california', 'diabetes']

model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRegressor(random_state=seed)
models = [model1, model2, model3, model4]


for i, data in enumerate(datasets):
    x = data.data
    y = data.target
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=seed,
        #  stratify=y
    )
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print("==================", dataset_name[i], "==================" )
    
    #2. 모델구성
    for model in models:
        model.fit(x_train, y_train)
        print("#################", model.__class__.__name__, "#################")
        print(' acc : ', model.score(x_test, y_test))
        print(model.feature_importances_)
    

