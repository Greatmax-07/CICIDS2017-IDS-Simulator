import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import time
from datetime import datetime

# === 1. Track Script Start Time === #
start_time = time.time()
print(f"[INFO] Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === 2. Load Cleaned Dataset === #
print("[INFO] Loading cleaned dataset...")
df = pd.read_csv('data/cleaned_dataset.csv')

# Shuffle dataset to remove day-wise bias while preserving label proportions
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# === 3. Load Pre-fitted Label Encoder === #
print("[INFO] Loading saved label encoder...")
label_encoder = joblib.load('outputs/label_encoder.pkl')
df['EncodedLabel'] = label_encoder.transform(df['Label'])

# === 4. Split Features and Target === #
X = df.drop(columns=['Label', 'Day', 'EncodedLabel'])    # Drop non-feature columns
y = df['EncodedLabel']                                   # Use encoded labels as target

# === 5. Split into train (70%), test (25%), demo (5%) with stratification === #
print("[INFO] Splitting dataset (70% train, 25% test, 5% demo)...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_test, X_demo, y_test, y_demo = train_test_split(X_temp, y_temp, test_size=(5/30), stratify=y_temp, random_state=42)

# === 6. Apply SMOTE to Address Class Imbalance in Training Set === #
print("[INFO] Applying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# === 7. Train XGBoost Model === #
print("[INFO] Training XGBoost model...")
xgb_clf = XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)
xgb_clf.fit(X_train_res, y_train_res)

# === 8. Add Labels Back to Dataset Splits === #
X_train_res['Label'] = label_encoder.inverse_transform(y_train_res)
X_test['Label'] = label_encoder.inverse_transform(y_test)
X_demo['Label'] = label_encoder.inverse_transform(y_demo)

# === 9. Save Train/Test/Demo Sets for Downstream Tasks === #
print("[INFO] Saving dataset splits...")
X_train_res.to_csv('outputs/train_data.csv', index=False)
X_test.to_csv('outputs/test_data.csv', index=False)
X_demo.to_csv('outputs/demo_data.csv', index=False)

# === 10. Save Model and Training Columns Only === #
print("[INFO] Saving model and training columns...")
joblib.dump(xgb_clf, 'outputs/xgb_model.pkl')
joblib.dump(list(X.columns), 'outputs/training_columns.pkl')

# === 11. Track Script End Time === #
end_time = time.time()
elapsed = round(end_time - start_time, 2)
print(f"\n[INFO] Final model and data saved successfully. Total time taken: {elapsed} seconds.")
