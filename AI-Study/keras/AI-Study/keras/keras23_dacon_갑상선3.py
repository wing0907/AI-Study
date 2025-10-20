import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
from keras.metrics import AUC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# === Load Data ===
path = './_data/dacon/갑상선암/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# === Separate Features and Target ===
x = train_csv.drop(['Cancer'], axis=1)
y = train_csv['Cancer']

# === Encode Categorical Columns ===
categorical_cols = x.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    test_csv[col] = le.transform(test_csv[col])

# === Scale Data ===
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
test_csv = scaler.transform(test_csv)

# === Stratified Split ===
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=55)
for train_idx, val_idx in sss.split(x, y):
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# === Class Weights ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_dict = dict(enumerate(class_weights))

# === Build Model ===
model = Sequential()
model.add(Dense(128, input_dim=x.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
# model.add(Dropout(0.2))

model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Dense(1, activation='sigmoid'))

# === Compile ===
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', AUC(name='auc')]
)


import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

path = './_save/keras23_cancer/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k23_', date, '_', filename])


# === Callbacks ===
es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', save_best_only=True, verbose=1, filepath=filepath)

# === Train ===
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=1000,
    batch_size=64,
    callbacks=[es, lr, mcp],
    verbose=1,
    class_weight=class_weights_dict
)

# === Evaluation ===
loss, accuracy, auc = model.evaluate(x_val, y_val, verbose=0)
y_val_pred = model.predict(x_val).ravel()

# === Threshold Tuning ===
best_threshold, best_f1 = 0.5, 0
for threshold in np.arange(0.3, 0.7, 0.01):
    preds = (y_val_pred > threshold).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f'✅ loss: {loss:.4f}')
print(f'✅ acc : {accuracy:.4f}')
print(f'✅ AUC : {auc:.4f}')
print(f'✅ Best F1: {best_f1:.4f} at threshold {best_threshold:.2f}')

# === Predict Test Set ===
y_submit = model.predict(test_csv).ravel()
submission_csv['Cancer'] = (y_submit > best_threshold).astype(int)
submission_csv.to_csv(path + 'submission_0609_best.csv')
print("✅ Submission saved!")


# print(train_csv.info())
# print(train_csv.shape)
# print(train_csv)        # [87159 rows x 15 columns]

# print(test_csv.info())
# # print(test_csv.shape)
# # print(test_csv)     #[46204 rows x 14 columns]

# print(train_csv.columns)
# #Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#     #    'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#     #    'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
#     #    'Cancer'],
#     #   dtype='object')
    
# print(test_csv.columns)
# Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#        'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#        'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result'],
#       dtype='object')



#  loss: 0.5638
# ✅ acc : 0.8795
# ✅ AUC : 0.6911
# ✅ Best F1: 0.4686 at threshold 0.50
# ✅ Submission saved!

# Epoch 00044: early stopping
# ✅ loss: 0.5607
# ✅ acc : 0.8762
# ✅ AUC : 0.7125
# ✅ Best F1: 0.4961 at threshold 0.66
# ✅ Submission saved!

# Epoch 00075: early stopping
# ✅ loss: 0.5485
# ✅ acc : 0.8862
# ✅ AUC : 0.7183
# ✅ Best F1: 0.5021 at threshold 0.45
# ✅ Submission saved!

# ✅ loss: 0.5480
# ✅ acc : 0.8798
# ✅ AUC : 0.6972
# ✅ Best F1: 0.4626 at threshold 0.45
# ✅ Submission saved!