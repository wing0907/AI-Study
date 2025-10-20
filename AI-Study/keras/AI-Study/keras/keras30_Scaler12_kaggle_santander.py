# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
# 이진분류

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
import time

#  1. data
path = './_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
print(train_csv)        # [200000 rows x 201 columns]
"""
              target    var_0    var_1    var_2   var_3    var_4    var_5   var_6  ...  var_192  var_193  var_194  var_195  var_196  var_197  var_198  var_199
ID_code                                                                            ...
train_0            0   8.9255  -6.7863  11.9081  5.0930  11.4607  -9.2834  5.1187  ...   3.1364   1.6910  18.5227  -2.3978   7.8784   8.5635  12.7803  -1.0914
train_1            0  11.5006  -4.1473  13.8588  5.3890  12.3622   7.0433  5.6208  ...   2.5837  10.9516  15.4305   2.0339   8.1267   8.7889  18.3560   1.9518
train_2            0   8.6093  -2.7457  12.0805  7.8928  10.5825  -9.0837  6.9427  ...   1.6704   1.6858  21.6042   3.1417  -6.5213   8.2675  14.7222   0.3965
train_3            0  11.0604  -2.1518   8.9522  7.1957  12.5846  -1.8361  5.8428  ...   0.7178   1.4214  23.0347  -1.2706  -2.9275  10.2922  17.9697  -8.9996
train_4            0   9.8369  -1.4834  12.8746  6.6375  12.2772   2.4486  5.9405  ...  -0.1508   9.1942  13.2876  -1.5121   3.9267   9.5031  17.9974  -8.8104
...              ...      ...      ...      ...     ...      ...      ...     ...  ...      ...      ...      ...      ...      ...      ...      ...      ...
train_199995       0  11.4880  -0.4956   8.2622  3.5142  10.3404  11.6081  5.6709  ...   3.9901   0.9388  18.0249  -1.7939   2.1661   8.5326  16.6660 -17.8661
train_199996       0   4.9149  -2.4484  16.7052  6.6345   8.3096 -10.5628  5.8802  ...   0.6998   1.8341  22.2717   1.7337  -2.1651   6.7419  15.9054   0.3388
train_199997       0  11.2232  -5.0518  10.5127  5.6456   9.3410  -5.4086  4.5555  ...   3.1032   4.8793  23.5311  -1.5736   1.2832   8.7155  13.8329   4.1995
train_199998       0   9.7148  -8.6098  13.6104  5.7930  12.5173   0.5339  6.0479  ...   2.7337  11.1178  20.4158  -0.0786   6.7980  10.0342  15.5289 -13.9001
train_199999       0  10.8762  -5.7105  12.1183  8.0328  11.5577   0.3488  5.2839  ...   0.1276   0.3766  15.2101  -2.4907  -2.2342   8.1857  12.1284   0.1385
"""

print(train_csv.isna().sum())   
print(test_csv.isna().sum())
print(train_csv.columns)
# Index(['target', 'var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6',
#        'var_7', 'var_8',
#        ...
#        'var_190', 'var_191', 'var_192', 'var_193', 'var_194', 'var_195',
#        'var_196', 'var_197', 'var_198', 'var_199'],
#       dtype='object', length=201)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']
print(x.shape, y.shape) # (200000, 200) (200000,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, shuffle=True, 
    random_state= 190,)

print(x_train)  # [160000 rows x 200 columns]
print(y_train)




scaler = MinMaxScaler()
scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# 2. modelling
model = Sequential([
    Dense(256, input_dim=200, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# 3. compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()
print(date)     # 2025-06-02 13:00:44.718308
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")     # string 문자열
print(date)     # 0602_1305
print(type(date))   # <class 'str'>

path = './_save/keras30_scaler/santander/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k30_', date, '_', filename])



mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=filepath
)



start_time = time.time()

hist = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=1000,
    batch_size=512,
    callbacks=[es, mcp],
    class_weight=class_weights,
    verbose=1
)
end_time = time.time()

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('Evaluation - Loss:', (loss, 4), 'Accuracy:', round(acc, 4))

# Predict on test split
y_pred_prob = model.predict(x_test)
threshold = 0.45
y_pred_binary = (y_pred_prob > threshold).astype(int)
acc_score = accuracy_score(y_test, y_pred_binary)
print(f"Threshold={threshold}, Accuracy={round(acc_score, 4)}")


print("acc_score : ", acc_score)
print("걸린시간 : ", round(end_time - start_time, 2), "초")   


# Predict on submission set
# 테스트셋도 스케일링 필요
test_scaled = scaler.transform(test_csv)
y_submit = model.predict(test_scaled)
y_submit = np.round(y_submit)


submission_csv['target'] = y_submit
submission_csv.to_csv(path + 'sample_submission_0609_0955.csv')
print("Submission saved as 'sample_submission_0609_0955.csv'")

import tensorflow as tf
# print(tf.__version__) # 2.9.3           tensorflow는 2.10 버전까지는 gpu와 cpu 버전이 따로 나누어져있다

gpus =tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')
    

# Evaluation - Loss: (0.24565181136131287, 4) Accuracy: 0.9098
# Threshold=0.45, Accuracy=0.9123
# acc_score :  0.912275
# 걸린시간 :  649.14 초
# Submission saved as 'sample_submission_0605_1755.csv'


# cpu       MinMaxScaler
# Evaluation - Loss: (0.2707967460155487, 4) Accuracy: 0.9147
# Threshold=0.45, Accuracy=0.9129
# acc_score :  0.9129
# 걸린시간 :  1410.06 초
# Submission saved as 'sample_submission_0609_0155.csv'

# Evaluation - Loss: (0.23841562867164612, 4) Accuracy: 0.9143
# Threshold=0.45, Accuracy=0.9156
# acc_score :  0.91555
# 걸린시간 :  621.62 초
# Submission saved as 'sample_submission_0609_0955.csv'


##############################################################
# gpu
# Evaluation - Loss: (0.2699181139469147, 4) Accuracy: 0.9001
# Threshold=0.45, Accuracy=0.894
# acc_score :  0.893975
# 걸린시간 :  263.98 초