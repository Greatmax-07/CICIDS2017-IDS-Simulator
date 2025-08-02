import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib

# === 1. Track Script Start Time === #
start_time = time.time()
print(f"[INFO] Data processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === 2. Load Excel File Containing Multiple Sheets === #
file_path = r"data\feature_selected_dataset.xlsx"
print(f"[INFO] Loading Excel file from: {file_path}")
xls = pd.ExcelFile(file_path)

# === 3. Initialize List to Store Processed DataFrames === #
dfs = []

# === 4. Read and Process Each Sheet Individually === #
print("[INFO] Reading and processing sheets...")
for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)  # Load data from current sheet

    # Strip extra spaces from column names to ensure consistency
    df.columns = df.columns.str.strip()

    # Extract the day name from the sheet name (e.g., "Thursday-Afternoon" -> "Thursday")
    day_name = sheet.split('-')[0]
    df["Day"] = day_name  # Add day as a new column

    # Append processed dataframe to list
    dfs.append(df)

# === 5. Combine All Sheets into a Single DataFrame === #
print("[INFO] Merging all sheets into one DataFrame...")
final_df = pd.concat(dfs, ignore_index=True)

# === 6. Clean and Encode Label Column === #
print("[INFO] Cleaning and encoding labels...")
# Encode string labels into numeric form
label_encoder = LabelEncoder()
final_df['LabelEncoded'] = label_encoder.fit_transform(final_df['Label'])


# === 7. Save Label Mapping and Label Encoder === #
encoder_path = "outputs/label_encoder.pkl"
joblib.dump(label_encoder, encoder_path)
print(f"[INFO] Label encoder saved to: {encoder_path}")

label_mapping_path = "outputs/label_mapping.json"
label_mapping = {str(k): int(v) for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

with open(label_mapping_path, "w") as f:
    f.write(json.dumps(label_mapping, indent=4))
print(f"[INFO] Label mapping saved to: {label_mapping_path}")


# === 8. Clean Dataset: Drop Blanks (Missing Values Only) === #
print("\n[INFO] Checking for missing, inf, and NaN values (for verification only)...")
print(final_df.isin([np.nan, np.inf, -np.inf]).sum())

# === 9. Clean Dataset: Drop Blanks (Missing Values Only) === #
print("\n[INFO] Label counts before dropping blanks:")
print(final_df['Label'].value_counts(dropna=False))

# Drop rows with blank values (in any column)
final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_df.dropna(inplace=True)
final_df.reset_index(drop=True, inplace=True)

print("\n[INFO] Label counts after dropping blanks:")
print(final_df['Label'].value_counts(dropna=False))

# === 10. Correlation Heatmap (After Cleaning) === #
print("[INFO] Generating correlation heatmap after cleaning...")

# Get the correlation matrix of numeric features only
corr_matrix = final_df.drop(columns=['LabelEncoded']).corr(numeric_only=True)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(20, 12))  # Bigger canvas
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 8}
)
plt.title("Feature Correlation Heatmap (After Cleaning)", fontsize=16)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap_after_cleaning.png")
plt.show()

# === 11. Final Dataset Info === #
print("\n[INFO] Final dataset structure:")
final_df.info()

print("\n[INFO] Rows per day:")
print(final_df["Day"].value_counts())

# === 12. Save Cleaned Dataset to CSV === #
cleaned_path = "data/cleaned_dataset.csv"
final_df.to_csv(cleaned_path, index=False)
print(f"\n[INFO] Full cleaned dataset saved successfully to: {cleaned_path}")

# === 13. Track Script End Time === #
end_time = time.time()
elapsed = round(end_time - start_time, 2)
print(f"\n[INFO] Data preparation completed. Total time taken: {elapsed} seconds.")
