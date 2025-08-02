import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from datetime import datetime
from report_generator import generate_model_report_pdf

start_time = time.time()
print(f"[INFO] Testing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === 1. Load Model, Demo Data, and Preprocessing Objects === #
print("[INFO] Loading model and test data...")
demo_data = pd.read_csv('outputs/demo_data.csv')
model = joblib.load('outputs/xgb_model.pkl')
label_encoder = joblib.load('outputs/label_encoder.pkl')
training_columns = joblib.load('outputs/training_columns.pkl')

# === 2. Determine Supported vs Unsupported Labels === #
print("[INFO] Analysing attack types in the dataset...")
supported_labels = set(label_encoder.classes_)
input_labels = set(demo_data['Label'].unique())
unsupported_labels = input_labels - supported_labels
valid_labels = input_labels & supported_labels

if not valid_labels:
    print("No known attack types found in demo data. Cannot perform meaningful evaluation.")
    exit()

if unsupported_labels:
    print(f"[INFO] Unsupported attack types found: {unsupported_labels}")
    print("[INFO] Proceeding with analysis of supported types only.\n")

# === 3. Filter demo data to only keep BENIGN and supported attack types === #
print("[INFO] Filtering the data...")
filtered_data = demo_data[demo_data['Label'].isin(valid_labels | {'BENIGN'})]

# === 4. Feature Extraction & Prediction === #
print("[INFO] Making predictions...")
X_test = filtered_data[training_columns]
y_test = label_encoder.transform(filtered_data['Label'])
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# === 5. Classification Report === #
print("[INFO] Generating classification report...")
present_labels = sorted(set(y_test))
present_class_names = label_encoder.inverse_transform(present_labels)

report = classification_report(y_test, y_pred, target_names=present_class_names)
print("\n=== CLASSIFICATION REPORT ===\n", report)

with open('outputs/demo_data_classification_report.txt', 'w') as f:
    f.write(report)

# === 6. Confusion Matrix === #
print("[INFO] Plotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=present_class_names, yticklabels=present_class_names,
            cbar_kws={"shrink": 0.8}, linewidths=0.5, linecolor='grey')
plt.title("Confusion Matrix of XGBoost Model Predictions", fontsize=16, weight='bold')
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig("outputs/demo_data_confusion_matrix.png", dpi=300)
plt.close()

# === 7. ROC Curve (Multiclass) === #
print("[INFO] Plotting ROC curves...")
y_test_bin = label_binarize(y_test, classes=present_labels)
fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(present_labels)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, present_labels[i]])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(12, 10))
colors = plt.colormaps.get_cmap('tab20')

for i, label in enumerate(present_class_names):
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
plt.savefig("outputs/demo_data_roc_multiclass.png", dpi=300)
plt.close()

# === 8. Completion Time === #
end_time = time.time()
print(f"\n[INFO] Testing finished in {end_time - start_time:.2f} seconds.")

# === 9. Report Generation === #
generate_model_report_pdf()