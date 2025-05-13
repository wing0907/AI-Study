import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
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
scaler = RobustScaler()
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

# 10. Callback
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)


# 11. Train
model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=8,
    callbacks=[es],
    class_weight=class_weights,
    verbose=1
)

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
# sample_submission_csv.to_csv(path + 'submission_0528_0930.csv')
# print("Submission saved as 'submission_0528_0930.csv'")

# scaler = MinMaxScaler()
# Evaluation - Loss: 0.4303 Accuracy: 0.7939
# Final Accuracy Score: 0.7939


# scaler = StandardScaler()
# Evaluation - Loss: 0.5066 Accuracy: 0.7481
# Final Accuracy Score: 0.7481


# scaler = MaxAbsScaler()
# Evaluation - Loss: 0.4531 Accuracy: 0.7634
# Final Accuracy Score: 0.7634


# scaler = RobustScaler()
# Evaluation - Loss: 0.4974 Accuracy: 0.7481
# Final Accuracy Score: 0.7481