# ============================================================
#  predict.py
#  Student Marks Prediction – Interactive Prediction
# ============================================================

import numpy as np
import joblib
import os

MODEL_PATH  = os.path.join("model", "linear_regression.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

FEATURES = ["StudyHours", "AttendancePercentage",
            "PreviousExamMarks", "AssignmentMarks", "InternalMarks"]

LIMITS = {
    "StudyHours":             (0.0, 24.0),
    "AttendancePercentage":   (0.0, 100.0),
    "PreviousExamMarks":      (0.0, 100.0),
    "AssignmentMarks":        (0.0, 25.0),
    "InternalMarks":          (0.0, 30.0),
}

def load_artefacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Please run train_model.py first."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def get_grade(marks):
    if marks >= 85:
        return "Excellent"
    elif marks >= 70:
        return "Good"
    elif marks >= 55:
        return "Average"
    elif marks >= 40:
        return "Below Average"
    else:
        return "Fail"

def get_float_input(prompt, low, high):
    while True:
        try:
            val = float(input(prompt))
            if low <= val <= high:
                return val
            print(f"   [!] Please enter a value between {low} and {high}.")
        except ValueError:
            print("   [!] Invalid input. Please enter a number.")

def predict_marks(model, scaler, input_values):
    """Predict final marks for a single student."""
    arr = np.array(input_values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)[0]
    # Clip to valid range
    prediction = max(0.0, min(100.0, prediction))
    return round(prediction, 2)

def run_interactive():
    print("=" * 55)
    print("   STUDENT MARKS PREDICTION - ENTER STUDENT DETAILS")
    print("=" * 55)

    try:
        model, scaler = load_artefacts()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    print("\nEnter the following details for the student:\n")

    inputs = []
    prompts = [
        ("StudyHours",           "   Study Hours per day        (0 - 24)  : "),
        ("AttendancePercentage", "   Attendance Percentage      (0 - 100) : "),
        ("PreviousExamMarks",    "   Previous Exam Marks        (0 - 100) : "),
        ("AssignmentMarks",      "   Assignment Marks           (0 - 25)  : "),
        ("InternalMarks",        "   Internal / Practical Marks (0 - 30)  : "),
    ]

    for feat, prompt in prompts:
        low, high = LIMITS[feat]
        val = get_float_input(prompt, low, high)
        inputs.append(val)

    predicted = predict_marks(model, scaler, inputs)
    grade     = get_grade(predicted)

    print("\n" + "=" * 55)
    print("   PREDICTION RESULT")
    print("=" * 55)
    print(f"\n   Predicted Final Marks : {predicted} / 100")
    print(f"   Grade Category        : {grade}")
    print()

    # Advice
    study_hours = inputs[0]
    attendance  = inputs[1]
    if study_hours < 3:
        print("   [Tip] Try to study at least 3-4 hours daily.")
    if attendance < 75:
        print("   [Tip] Attendance below 75% may affect eligibility.")

    print("=" * 55 + "\n")

if __name__ == "__main__":
    run_interactive()
