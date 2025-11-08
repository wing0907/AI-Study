import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
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

# # 8. Model
# model = Sequential()
# model.add(Dense(128, input_dim=8, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(1, activation='sigmoid'))

input1 = Input(shape=(8,))
dense1 = Dense(128, activation='relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(64, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense2)
output1 = Dense(1, activation='sigmoid')(drop2)
model2 = Model(inputs=input1, outputs=output1)
model2.summary()