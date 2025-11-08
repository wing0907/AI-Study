import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score

# 1. Load Data
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Split x and y
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

# 3. Replace 0s with NaN (only in specific columns)
zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_not_allowed] = x[zero_not_allowed].replace(0, np.nan)

# 4. Fill NaNs with mean
x = x.fillna(x.mean())

# 5. Scale Data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_csv)

# 6. Split train/test
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size=0.8, random_state=33
)

# 7. Compute class weights
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# 8. Model
model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# 9. Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

import time
start = time.time()
hist = model.fit(x_train,y_train, epochs = 100, batch_size =32,
          verbose = 1,
          validation_split = 0.2,       
          )
end = time.time()


import tensorflow as tf
gpus =tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')

# 12. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print('Evaluation - Loss:', round(loss, 4), 'Accuracy:', round(acc, 4))

# 13. Predict on test split
y_pred_prob = model.predict(x_test)
y_pred_binary = (y_pred_prob > 0.5).astype(int)
acc_score = accuracy_score(y_test, y_pred_binary)
print('Final Accuracy Score:', round(acc_score, 4))

# 14. Predict on submission set
y_submit = model.predict(test_scaled)
y_submit = np.round(y_submit)

sample_submission_csv['Outcome'] = y_submit
print("걸린시간 :", end - start)

# cpu 사용결과
# Evaluation - Loss: 0.6658 Accuracy: 0.7405
# Final Accuracy Score: 0.7405
# 걸린시간 : 3.143235445022583


# gpu 사용결과
# Evaluation - Loss: 0.615 Accuracy: 0.7557
# Final Accuracy Score: 0.7557
# 걸린시간 : 6.656480550765991