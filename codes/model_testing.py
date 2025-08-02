import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from datetime import datetime

start_time = time.time()
print(f"[INFO] Testing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === 1. Load Model, Test Data, and Preprocessing Objects === #
print("[INFO] Loading model and test data...")
test_data = pd.read_csv('outputs/test_data.csv')
model = joblib.load('outputs/xgb_model.pkl')
label_encoder = joblib.load('outputs/label_encoder.pkl')
training_columns = joblib.load('outputs/training_columns.pkl')

# === 2. Feature Extraction & Prediction === #
print("[INFO] Making predictions...")
X_test = test_data[training_columns]
y_test = label_encoder.transform(test_data['Label'])
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# === 3. Classification Report === #
print("[INFO] Generating classification report...")
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("\n=== CLASSIFICATION REPORT ===\n", report)
with open('outputs/test_data_classification_report.txt', 'w') as f:
    f.write(report)

# === 4. Confusion Matrix === #
print("[INFO] Plotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
            cbar_kws={"shrink": 0.8}, linewidths=0.5, linecolor='grey')

plt.title("Confusion Matrix of XGBoost Model Predictions", fontsize=16, weight='bold')
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig("outputs/test_data_confusion_matrix.png", dpi=300)
plt.close()


# === 5. ROC Curve (Multiclass) === #
print("[INFO] Plotting ROC curves...")
y_test_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))
fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(label_encoder.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(12, 10))
colors = plt.colormaps.get_cmap('tab20')
for i, label in enumerate(label_encoder.classes_):
    plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
             label=f'{label} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance level (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Multiclass ROC Curve\n(XGBoost Model Performance for Each Attack Type)', fontsize=16, weight='bold')
plt.legend(loc='lower right', fontsize='small')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/test_data_roc_multiclass.png", dpi=300)
plt.close()

# === 6. Completion Time === #
end_time = time.time()
print(f"\n[INFO] Testing finished in {end_time - start_time:.2f} seconds.")
