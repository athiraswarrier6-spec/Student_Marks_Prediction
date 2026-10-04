# ============================================================
#  main.py
#  Student Marks Prediction – Main Entry Point
#  IBM SkillsBuild Machine Learning & Applied AI Internship
# ============================================================
#
#  Run this file to execute the complete pipeline:
#    python main.py
#
#  Individual scripts can also be run separately:
#    python explore_data.py
#    python train_model.py
#    python visualize.py
#    python predict.py
# ============================================================

import sys

def print_banner():
    print()
    print("=" * 57)
    print("  STUDENT MARKS PREDICTION USING MACHINE LEARNING")
    print("       IBM SkillsBuild Internship Project")
    print("=" * 57)
    print()

def step_header(num, title):
    print(f"\n{'-' * 55}")
    print(f"  STEP {num}: {title}")
    print(f"{'-' * 55}")

def main():
    print_banner()

    # ── STEP 1: Data Exploration ─────────────────────────────
    step_header(1, "DATA EXPLORATION")
    try:
        import explore_data   # noqa: F401 – side-effect import (prints)
    except Exception as e:
        print(f"[ERROR] Data exploration failed: {e}")
        sys.exit(1)

    # ── STEP 2: Model Training & Evaluation ──────────────────
    step_header(2, "MODEL TRAINING & EVALUATION")
    try:
        from train_model import main as train_main
        model, scaler, X_test, y_test, y_pred, metrics = train_main()
    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        sys.exit(1)

    # ── STEP 3: Visualisations ────────────────────────────────
    step_header(3, "GENERATING VISUALISATIONS")
    try:
        from visualize import generate_all
        generate_all(y_test=y_test, y_pred=y_pred, metrics=metrics)
    except Exception as e:
        print(f"[ERROR] Visualisation failed: {e}")
        sys.exit(1)

    # ── STEP 4: Interactive Prediction ───────────────────────
    step_header(4, "INTERACTIVE PREDICTION")
    print("\n  Would you like to predict marks for a new student?")
    choice = input("  Enter 'y' to continue or any other key to skip: ").strip().lower()
    if choice == "y":
        try:
            from predict import run_interactive
            run_interactive()
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PROJECT PIPELINE COMPLETE")
    print("=" * 55)
    print(f"\n  Final Model Metrics:")
    print(f"     MAE  : {metrics['MAE']:.4f}")
    print(f"     MSE  : {metrics['MSE']:.4f}")
    print(f"     RMSE : {metrics['RMSE']:.4f}")
    print(f"     R2   : {metrics['R2']:.4f}")
    print(f"\n  Outputs:")
    print(f"     - Trained model  -> model/linear_regression.pkl")
    print(f"     - Scaler         -> model/scaler.pkl")
    print(f"     - Plots          -> plots/  (6 PNG files)")
    print()

if __name__ == "__main__":
    main()
