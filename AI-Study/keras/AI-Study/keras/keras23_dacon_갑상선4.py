import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
from keras.metrics import AUC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import datetime

# === Load Data ===
path = './_data/dacon/갑상선암/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# === Encode Categorical Columns ===
categorical_cols = x.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    test_csv[col] = le.transform(test_csv[col])

# === Robust Scaling ===
scaler = RobustScaler()
x = scaler.fit_transform(x)
test_csv = scaler.transform(test_csv)

# === Stratified Split ===
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=27)
for train_idx, val_idx in sss.split(x, y):
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# === Class Weights ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_dict = dict(enumerate(class_weights))

# === Neural Network Model ===
model = Sequential()
model.add(Dense(128, input_dim=x.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', AUC(name='auc')]
)

# === Callbacks ===
date = datetime.datetime.now().strftime("%m%d_%H%M")
path_save = './_save/keras23_cancer/'
filepath = path_save + f'k23_{date}_' + '{epoch:04d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', save_best_only=True, verbose=1, filepath=filepath)

# === Train NN ===
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1000,
    batch_size=64,
    callbacks=[es, lr, mcp],
    verbose=1,
    class_weight=class_weights_dict
)

# === XGBoost 앙상블 ===
xgb_model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(x_train, y_train)

# === Validation Ensemble ===
nn_val = model.predict(x_val).ravel()
xgb_val = xgb_model.predict_proba(x_val)[:, 1]
ensemble_val = (nn_val + xgb_val) / 2

# === Threshold Tuning ===
best_threshold, best_f1 = 0.5, 0
for threshold in np.arange(0.3, 0.7, 0.01):
    preds = (ensemble_val > threshold).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f'✅ Best F1: {best_f1:.4f} at threshold {best_threshold:.2f}')

# === Test Prediction ===
nn_test = model.predict(test_csv).ravel()
xgb_test = xgb_model.predict_proba(test_csv)[:, 1]
ensemble_test = (nn_test + xgb_test) / 2
submission_csv['Cancer'] = (ensemble_test > best_threshold).astype(int)
submission_csv.to_csv(path + f'submission_{date}_ensemble.csv')
print("✅ Submission saved!")


# Epoch 00084: early stopping
# ✅ Best F1: 0.5094 at threshold 0.30