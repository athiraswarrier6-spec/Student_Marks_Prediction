# ============================================================
#  train_model.py
#  Student Marks Prediction – Model Training & Evaluation
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score)
from sklearn.preprocessing import StandardScaler

# ── Configuration ────────────────────────────────────────────
FEATURES   = ["StudyHours", "AttendancePercentage",
               "PreviousExamMarks", "AssignmentMarks", "InternalMarks"]
TARGET     = "FinalMarks"
TEST_SIZE  = 0.2
RANDOM_STATE = 42
MODEL_DIR  = "model"

# ── Load dataset ─────────────────────────────────────────────
def load_data(path="student_data.csv"):
    df = pd.read_csv(path)
    return df

# ── Preprocess ───────────────────────────────────────────────
def preprocess(df):
    """Drop unused columns, handle missing values, return X and y."""
    df = df.drop(columns=["StudentID", "Grade"], errors="ignore")
    df = df.dropna()
    X = df[FEATURES]
    y = df[TARGET]
    return X, y

# ── Train ────────────────────────────────────────────────────
def train(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ── Evaluate ─────────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    return y_pred, {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# ── Save artefacts ───────────────────────────────────────────
def save_artefacts(model, scaler):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  os.path.join(MODEL_DIR, "linear_regression.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"   [OK] Model and scaler saved to '{MODEL_DIR}/' folder.")

# ── Main pipeline ────────────────────────────────────────────
def main():
    print("=" * 55)
    print("   STUDENT MARKS PREDICTION - MODEL TRAINING")
    print("=" * 55)

    # 1. Load
    df = load_data()
    print(f"\n[INFO] Dataset loaded: {len(df)} records, {len(FEATURES)} features.")

    # 2. Preprocess
    X, y = preprocess(df)

    # 3. Scale features
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 5. Train
    model = train(X_train, y_train)
    print("\n[INFO] Linear Regression model trained successfully.")

    # 6. Evaluate
    y_pred, metrics = evaluate(model, X_test, y_test)

    print("\n[INFO] Model Evaluation Metrics:")
    print(f"   Mean Absolute Error  (MAE)  : {metrics['MAE']:.4f}")
    print(f"   Mean Squared Error   (MSE)  : {metrics['MSE']:.4f}")
    print(f"   Root Mean Sq. Error  (RMSE) : {metrics['RMSE']:.4f}")
    print(f"   R2 Score             (R2)   : {metrics['R2']:.4f}")

    # 7. Feature importance (coefficients)
    print("\n[INFO] Feature Coefficients (after scaling):")
    for feat, coef in zip(FEATURES, model.coef_):
        print(f"   {feat:<25}: {coef:.4f}")
    print(f"   {'Intercept':<25}: {model.intercept_:.4f}")

    # 8. Sample predictions vs actuals
    print("\n[INFO] Sample Predictions vs Actual (first 10 test records):")
    print(f"   {'Actual':<10} {'Predicted':<12} {'Difference'}")
    print("   " + "-" * 35)
    for actual, pred in zip(list(y_test)[:10], y_pred[:10]):
        diff = abs(actual - pred)
        print(f"   {actual:<10.1f} {pred:<12.2f} {diff:.2f}")

    # 9. Save
    save_artefacts(model, scaler)

    # Return for use in other modules
    return model, scaler, X_test, y_test, y_pred, metrics

if __name__ == "__main__":
    main()
