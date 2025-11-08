import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
import tensorflow as tf

# 1. Load Data
path = './_data/dacon/갑상선암/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# 2. Label Encoding
cat_cols = ['Gender', 'Country', 'Race', 'Family_Background',
            'Radiation_History', 'Iodine_Deficiency', 'Smoke',
            'Weight_Risk', 'Diabetes']
for col in cat_cols:
    le = LabelEncoder()
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])

# 3. Split X / y
x = train_csv.drop('Cancer', axis=1)
y = train_csv['Cancer']

# 4. Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
test_scaled = scaler.transform(test_csv)

# 5. Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, train_size=0.85, random_state=34, stratify=y
)

# 6. Class Weights
weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))

# 7. Build Model
model = Sequential([
    Dense(64, input_dim=x_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

# 8. Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# 9. EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# 10. Train
model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=1000,
    batch_size=128,
    class_weight=class_weights,
    callbacks=[es],
    verbose=1
)

# 11. Predict & Threshold Tuning
y_prob = model.predict(x_test)
prec, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1s = 2 * prec * recall / (prec + recall + 1e-8)
best_thresh = thresholds[np.argmax(f1s)]
print("🔍 Best Threshold for F1:", best_thresh)

# Final Prediction with best threshold
y_pred = (y_prob > best_thresh).astype(int)

# 12. Metrics
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Final Accuracy: {acc:.4f}")
print(f"🎯 Final F1 Score: {f1:.4f}")

# 13. Submission
y_submit_prob = model.predict(test_scaled)
y_submit = (y_submit_prob > best_thresh).astype(int)
submission_csv['Cancer'] = y_submit
submission_csv.to_csv(path + 'submission_f1_opt2.csv')
print("📁 Submission saved as 'submission_f1_opt2.csv'")


# 🔍 Best Threshold for F1: 0.58693236

# ✅ Final Accuracy: 0.8806
# 🎯 Final F1 Score: 0.4717
# 📁 Submission saved as 'submission_f1_opt.csv'


# 🔍 Best Threshold for F1: 0.6565702

# ✅ Final Accuracy: 0.8818
# 🎯 Final F1 Score: 0.4741
# 📁 Submission saved as 'submission_f1_opt1.csv'


# 🔍 Best Threshold for F1: 0.5970513

# ✅ Final Accuracy: 0.8824
# 🎯 Final F1 Score: 0.4760
# 📁 Submission saved as 'submission_f1_opt2.csv'

# 🔍 Best Threshold for F1: 0.69552743

# ✅ Final Accuracy: 0.8870
# 🎯 Final F1 Score: 0.4983
# 📁 Submission saved as 'submission_f1_opt2.csv'