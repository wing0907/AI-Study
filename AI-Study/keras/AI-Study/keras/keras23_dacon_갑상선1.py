# https://dacon.io/competitions/official/236488/mysubmission

import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import BatchNormalization
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
path = './_data/dacon/갑상선암/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)
print(train_csv)        # [87159 rows x 15 columns]
print(test_csv)         # [46204 rows x 14 columns]
print(submission_csv)   # [46204 rows x 1 columns]
print(train_csv.columns)
"""
Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
       'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
       'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
       'Cancer']
"""
print(train_csv.isnull().sum()) # 결측치 없음
print(test_csv.isna().sum())    # 결측치 없음

cols = ['Gender', 'Country', 'Race', 'Family_Background',
       'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
       'Diabetes']
for col in cols:
    le =LabelEncoder()
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])

x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']
print(x)                # [87159 rows x 14 columns]
print(x.shape, y.shape)          # (87159, 14), (87159,)

print(np.unique(y, return_counts=True)) # (array([0, 1], dtype=int64), array([76700, 10459], dtype=int64))


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_csv)


# 6. Split train/test
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size=0.85, random_state=190
)


# 7. Compute class weights
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# 8. Model
model = Sequential([
    Dense(128, input_dim=x_train.shape[1], activation='relu'),
    # BatchNormalization(),
    
    Dense(256, activation='relu'),
    # BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    # BatchNormalization(),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

# 9. Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)
# 10. Callback
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    restore_best_weights=True
)

# 11. Train
model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=1000,
    batch_size=128,
    callbacks=[es],
    class_weight=class_weights,
    verbose=1
)

# 12. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print('Evaluation - Loss:', round(loss, 4), 'Accuracy:', round(acc, 4))

# 13. Predict on test split
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.48).astype(int)

f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)
acc_score = accuracy_score(y_test, y_pred)
print('Final F1 Score:', round(f1, 4))


# 14. Predict on submission set
y_submit = model.predict(test_scaled)
y_submit = np.round(y_submit)
submission_csv['Cancer'] = y_submit
submission_csv.to_csv(path + 'submission_0604_0930.csv')
print("Submission saved as 'submission_0604_0930.csv'")

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras23_cancer/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k23_', date, '_', filename])

# Evaluation - Loss: 0.5183 Accuracy: 0.8606        33
# F1 Score: 0.4452054794520548
# Final F1 Score: 0.4452
# Submission saved as 'submission_0529_1330.csv'

# Evaluation - Loss: 0.5255 Accuracy: 0.8334        42
# F1 Score: 0.42924528301886794
# Final F1 Score: 0.4292
# Submission saved as 'submission_0529_1330.csv'