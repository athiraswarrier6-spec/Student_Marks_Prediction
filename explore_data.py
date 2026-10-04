# ============================================================
#  explore_data.py
#  Student Marks Prediction – Data Exploration
# ============================================================

import pandas as pd
import numpy as np

# ── Load dataset ────────────────────────────────────────────
df = pd.read_csv("student_data.csv")

print("=" * 55)
print("   STUDENT MARKS PREDICTION - DATA EXPLORATION")
print("=" * 55)

# ── Basic info ───────────────────────────────────────────────
print("\n[INFO] First 5 rows of the dataset:")
print(df.head())

print("\n[INFO] Dataset Shape:", df.shape)
print("   Rows:", df.shape[0], "| Columns:", df.shape[1])

print("\n[INFO] Column Names:")
for col in df.columns:
    print(f"   - {col}")

print("\n[INFO] Data Types:")
print(df.dtypes)

# ── Missing values ───────────────────────────────────────────
print("\n[INFO] Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   No missing values found.")
else:
    print(missing[missing > 0])

# ── Descriptive statistics ───────────────────────────────────
print("\n[INFO] Descriptive Statistics:")
print(df.describe().round(2))

# ── Correlation with FinalMarks ──────────────────────────────
print("\n[INFO] Correlation with Final Marks:")
features = ["StudyHours", "AttendancePercentage",
            "PreviousExamMarks", "AssignmentMarks", "InternalMarks"]
corr = df[features + ["FinalMarks"]].corr()["FinalMarks"].drop("FinalMarks")
for feat, val in corr.items():
    print(f"   {feat:<25}: {val:.4f}")

# ── Class distribution insight ───────────────────────────────
print("\n[INFO] Final Marks Distribution (buckets):")
bins   = [0, 40, 55, 70, 85, 100]
labels = ["Fail (<40)", "Below Avg (40-55)", "Average (55-70)",
          "Good (70-85)", "Excellent (85-100)"]
df["Grade"] = pd.cut(df["FinalMarks"], bins=bins, labels=labels,
                     include_lowest=True)
print(df["Grade"].value_counts().sort_index())

print("\n[DONE] Exploration complete.\n")
