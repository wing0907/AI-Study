import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import time

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=707, train_size=0.8,
)
print(x_train.shape, y_train.shape) # (353, 10) (353,)

# 2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=128, node2=64, node3=32, node4=16, node5=8, lr=0.001):
    inputs = Input(shape=(10, ), name='inputs')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    x = Dense(node5, activation=activation, name='hidden5')(x)
    outputs = Dense(1, activation='linear', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['mse'], loss='mse')
    return model

def create_hyperparameter():
    batchs =  [32, 16, 8, 1, 64]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropouts = [0.2, 0.3, 0.4, 0.5]
    activations = ['relu', 'elu', 'selu', 'linear']
    node1 = [128, 64, 32, 16]
    node2 = [128, 64, 32, 16]
    node3 = [128, 64, 32, 16]
    node4 = [128, 64, 32, 16]
    node5 = [128, 64, 32, 16]
    return {
        'batch_size' : batchs,
        'optimizer' : optimizers,
        'drop' : dropouts,
        'activation' : activations,
        'node1' : node1,
        'node2' : node2,
        'node3' : node3,
        'node4' : node4,
        'node5' : node5,
        }



hyperparameters = create_hyperparameter()
print(hyperparameters)

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

keras_model = KerasRegressor(build_fn=build_model, verbose=1)

model = RandomizedSearchCV(keras_model, hyperparameters, cv=3,  # default 5
                           n_iter=2,        # default 10
                           verbose=1,
                           )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 콜백 설정
es  = EarlyStopping(monitor='val_loss', mode='auto', patience=20,
                    verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint('best_model.h5', monitor='mse',
                      mode='max', save_best_only=True, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', mode='min',
                        factor=0.8, patience=10, verbose=1)


# 3. 훈련
start = time.time()
model.fit(x_train, y_train, epochs=200, callbacks=[es, mcp, rlr], verbose=10,
          validation_split=0.1)
end = time.time()
# Fitting 5 folds for each of 10 candidates, totalling 50 fits

print('최적의 매개변수 : ', model.best_estimator_) # 최적의 파라미터가 출력 됨 (지정한 것 중에 제일 좋은 것)
print('최적의 매개변수 : ', model.best_params_) # 최적의 파라미터가 출력 됨 (전체 중에 제일 좋은 것)

# 4. 평가, 예측
print('best_score : ', model.best_score_)    # train에서 가장 좋은 score 가 나옴

print('model.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2_score : ', r2)
print('time : ', round(end - start, 4), 'seconds')

# r2_score :  0.21485323082666907
# time :  7.2587 seconds