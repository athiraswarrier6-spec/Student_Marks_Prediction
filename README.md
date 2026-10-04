# Student Marks Prediction Using Machine Learning

> **IBM SkillsBuild Machine Learning & Applied AI Internship Project**

---

## 📌 Objective

Predict a student's **final examination marks** using key academic performance indicators — study habits, attendance, past performance, and internal assessments — through a **Linear Regression** model.

---

## 📂 Project Structure

```
Student_Marks_Prediction/
│
├── student_data.csv        # Dataset (100 student records)
├── main.py                 # ▶ Run this to execute the full pipeline
│
├── explore_data.py         # Step 1 – Load & explore the dataset
├── train_model.py          # Step 2 – Preprocess, train, and evaluate
├── visualize.py            # Step 3 – Generate all plots
├── predict.py              # Step 4 – Interactive prediction for new students
│
├── model/                  # Auto-created after training
│   ├── linear_regression.pkl
│   └── scaler.pkl
│
├── plots/                  # Auto-created after visualisation
│   ├── 1_distribution.png
│   ├── 2_feature_vs_target.png
│   ├── 3_correlation_heatmap.png
│   ├── 4_actual_vs_predicted.png
│   ├── 5_metrics.png
│   └── 6_grade_distribution.png
│
└── README.md
```

---

## 📊 Dataset

**File:** `student_data.csv`  
**Records:** 100 students  
**Source:** Synthetically generated for educational purposes

| Column | Description | Range |
|---|---|---|
| `StudentID` | Unique identifier | 1 – 100 |
| `StudyHours` | Average daily study hours | 0 – 24 |
| `AttendancePercentage` | Class attendance (%) | 0 – 100 |
| `PreviousExamMarks` | Marks scored in the previous exam | 0 – 100 |
| `AssignmentMarks` | Total assignment marks obtained | 0 – 25 |
| `InternalMarks` | Internal / practical assessment marks | 0 – 30 |
| `FinalMarks` | ✅ **Target variable** – final exam marks | 0 – 100 |

---

## 🛠 Technologies Used

| Tool / Library | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **Pandas** | Data loading, cleaning, and manipulation |
| **NumPy** | Numerical computations |
| **Matplotlib** | Base plotting library |
| **Seaborn** | Statistical data visualisation |
| **Scikit-learn** | Machine learning (model, scaler, metrics) |
| **Joblib** | Saving and loading the trained model |

---

## ⚙️ Methodology

```
Raw CSV Data
     │
     ▼
1. Data Exploration  ──── shape, dtypes, statistics, correlations
     │
     ▼
2. Preprocessing     ──── drop unused cols → handle nulls → StandardScaler
     │
     ▼
3. Train / Test Split  ── 80 % training | 20 % testing (random_state=42)
     │
     ▼
4. Linear Regression ──── fit on training set
     │
     ▼
5. Evaluation        ──── MAE · MSE · RMSE · R² on test set
     │
     ▼
6. Visualisation     ──── 6 PNG charts saved to plots/
     │
     ▼
7. Prediction        ──── user inputs → scaled → model → predicted mark
```

**Why Linear Regression?**  
The features (study hours, attendance, etc.) have a near-linear relationship with final marks, making Linear Regression an ideal, interpretable starting model for a beginner-level project.

---

## 🚀 How to Run

### 1. Prerequisites

Make sure you have Python 3.8 or higher installed.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 3. Run the complete pipeline

```bash
python main.py
```

This single command:
- Explores and describes the dataset
- Trains and evaluates the model
- Generates all visualisation plots
- Optionally lets you predict marks for a new student

### 4. Run individual scripts

```bash
python explore_data.py   # Explore data only
python train_model.py    # Train & evaluate only
python visualize.py      # Generate plots only (no model needed)
python predict.py        # Interactive prediction (model must exist)
```

---

## 📈 Expected Output

### Console Output (excerpt)

```
╔═════════════════════════════════════════════════════╗
║   STUDENT MARKS PREDICTION USING MACHINE LEARNING  ║
║          IBM SkillsBuild Internship Project         ║
╚═════════════════════════════════════════════════════╝

── STEP 1: DATA EXPLORATION ─────────────────────────
  Dataset loaded: 100 records, 5 features.
  No missing values found.

── STEP 2: MODEL TRAINING & EVALUATION ──────────────
  Train size: 80 | Test size: 20

  📊 Model Evaluation Metrics:
     MAE  : ~0.5
     MSE  : ~0.5
     RMSE : ~0.7
     R²   : ~0.99
```

### Plots Generated

| Plot | Description |
|---|---|
| `1_distribution.png` | Histogram + box plot of Final Marks |
| `2_feature_vs_target.png` | Scatter plots: each feature vs Final Marks |
| `3_correlation_heatmap.png` | Correlation matrix of all features |
| `4_actual_vs_predicted.png` | Model accuracy scatter + residual plot |
| `5_metrics.png` | Bar chart of error metrics + R² gauge |
| `6_grade_distribution.png` | Pie chart of student grade categories |

### Interactive Prediction

```
Enter the following details for the student:

   📚 Study Hours per day        (0 – 24)  : 6
   🏫 Attendance Percentage      (0 – 100) : 90
   📝 Previous Exam Marks        (0 – 100) : 80
   📋 Assignment Marks           (0 – 25)  : 22
   🔬 Internal / Practical Marks (0 – 30)  : 27

═══════════════════════════════════════════════════════
   📊 PREDICTION RESULT
═══════════════════════════════════════════════════════

   Predicted Final Marks : 86.43 / 100
   Grade Category        : Excellent 🏆
```

---

## 📉 Grade Categories

| Marks Range | Grade |
|---|---|
| 85 – 100 | Excellent 🏆 |
| 70 – 84 | Good 👍 |
| 55 – 69 | Average 📘 |
| 40 – 54 | Below Average ⚠️ |
| 0 – 39 | Fail ❌ |

---

## 🔍 Key Findings

- All five features show a **strong positive correlation** with Final Marks (r > 0.95).
- **Previous Exam Marks** and **Internal Marks** are the strongest predictors.
- The model achieves an **R² score close to 0.99**, indicating excellent fit on this dataset.
- Residuals are randomly distributed around zero, confirming the linear assumption holds.

---

## 🎓 Learning Outcomes

After completing this project, you will understand:

- How to load and explore a CSV dataset with Pandas
- How to preprocess data (scaling, splitting)
- How to train a Linear Regression model with Scikit-learn
- How to evaluate a model using MAE, MSE, RMSE, and R²
- How to visualise data and results with Matplotlib and Seaborn
- How to save a trained model and use it for live predictions

---

## 👤 Author

**IBM SkillsBuild Machine Learning & Applied AI Internship**  
Project: Student Marks Prediction Using Machine Learning
