# ============================================================
#  visualize.py
#  Student Marks Prediction – Data & Results Visualisation
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for VS Code)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── Style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
PLOT_DIR = "plots"
FEATURES = ["StudyHours", "AttendancePercentage",
            "PreviousExamMarks", "AssignmentMarks", "InternalMarks"]

def ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────
# 1. Distribution of Final Marks
# ────────────────────────────────────────────────────────────
def plot_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Distribution of Final Marks", fontsize=14, fontweight="bold")

    sns.histplot(df["FinalMarks"], bins=15, kde=True,
                 color="steelblue", ax=axes[0])
    axes[0].set_title("Histogram with KDE")
    axes[0].set_xlabel("Final Marks")

    sns.boxplot(x=df["FinalMarks"], color="lightcoral", ax=axes[1])
    axes[1].set_title("Box Plot")
    axes[1].set_xlabel("Final Marks")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "1_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"   [OK] Saved: {path}")

# ────────────────────────────────────────────────────────────
# 2. Feature vs Final Marks scatter plots
# ────────────────────────────────────────────────────────────
def plot_feature_vs_target(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Feature vs Final Marks", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    colors = ["#3b82d4", "#e07b39", "#2ca02c", "#d62728", "#9467bd"]
    for i, (feat, color) in enumerate(zip(FEATURES, colors)):
        axes[i].scatter(df[feat], df["FinalMarks"],
                        alpha=0.6, color=color, edgecolors="white", s=50)
        # trend line
        m, b = np.polyfit(df[feat], df["FinalMarks"], 1)
        x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
        axes[i].plot(x_line, m * x_line + b, color="black",
                     linewidth=1.5, linestyle="--")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Final Marks")
        axes[i].set_title(f"{feat} vs Final Marks")

    axes[5].axis("off")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "2_feature_vs_target.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"   [OK] Saved: {path}")

# ────────────────────────────────────────────────────────────
# 3. Correlation heatmap
# ────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[FEATURES + ["FinalMarks"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", linewidths=0.5,
                annot_kws={"size": 10}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "3_correlation_heatmap.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"   [OK] Saved: {path}")

# ────────────────────────────────────────────────────────────
# 4. Actual vs Predicted marks
# ────────────────────────────────────────────────────────────
def plot_actual_vs_predicted(y_test, y_pred):
    y_test_arr = np.array(y_test)
    fig, axes  = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Predictions vs Actual Marks",
                 fontsize=14, fontweight="bold")

    # Scatter
    axes[0].scatter(y_test_arr, y_pred, alpha=0.7,
                    color="#3b82d4", edgecolors="white", s=60)
    lims = [min(y_test_arr.min(), y_pred.min()) - 2,
            max(y_test_arr.max(), y_pred.max()) + 2]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
    axes[0].set_xlabel("Actual Marks")
    axes[0].set_ylabel("Predicted Marks")
    axes[0].set_title("Actual vs Predicted")
    axes[0].legend()

    # Residuals
    residuals = y_test_arr - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.7,
                    color="#e07b39", edgecolors="white", s=60)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Predicted Marks")
    axes[1].set_ylabel("Residuals (Actual − Predicted)")
    axes[1].set_title("Residual Plot")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "4_actual_vs_predicted.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"   [OK] Saved: {path}")

# ────────────────────────────────────────────────────────────
# 5. Model evaluation metrics bar chart
# ────────────────────────────────────────────────────────────
def plot_metrics(metrics):
    display_metrics = {k: v for k, v in metrics.items() if k != "R2"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Model Evaluation Metrics", fontsize=14, fontweight="bold")

    # Error metrics
    names  = list(display_metrics.keys())
    values = list(display_metrics.values())
    colors = ["#3b82d4", "#e07b39", "#2ca02c"]
    bars   = axes[0].bar(names, values, color=colors, edgecolor="white",
                         width=0.5)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    axes[0].set_title("Error Metrics (lower is better)")
    axes[0].set_ylabel("Error Value")

    # R² gauge
    r2    = metrics["R2"]
    gauge = [r2, 1 - r2]
    clrs  = ["#2ca02c", "#d9d9d9"]
    wedge_props = dict(width=0.4)
    axes[1].pie(gauge, colors=clrs, wedgeprops=wedge_props,
                startangle=90, counterclock=False)
    axes[1].text(0, 0, f"R² = {r2:.4f}", ha="center", va="center",
                 fontsize=14, fontweight="bold")
    axes[1].set_title("R² Score (closer to 1 is better)")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "5_metrics.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"   [OK] Saved: {path}")

# ────────────────────────────────────────────────────────────
# 6. Grade category pie chart
# ────────────────────────────────────────────────────────────
def plot_grade_distribution(df):
    bins   = [0, 40, 55, 70, 85, 100]
    labels = ["Fail\n(<40)", "Below Avg\n(40-55)",
              "Average\n(55-70)", "Good\n(70-85)", "Excellent\n(85-100)"]
    df["Grade"] = pd.cut(df["FinalMarks"], bins=bins,
                         labels=labels, include_lowest=True)
    counts = df["Grade"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors  = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=140,
        pctdistance=0.75, labeldistance=1.1
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title("Student Grade Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "6_grade_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"   [OK] Saved: {path}")

# ────────────────────────────────────────────────────────────
# Master function called by main.py
# ────────────────────────────────────────────────────────────
def generate_all(y_test=None, y_pred=None, metrics=None):
    ensure_plot_dir()
    df = pd.read_csv("student_data.csv")

    print("\n[INFO] Generating visualisations ...")
    plot_distribution(df)
    plot_feature_vs_target(df)
    plot_correlation_heatmap(df)
    if y_test is not None and y_pred is not None:
        plot_actual_vs_predicted(y_test, y_pred)
    if metrics is not None:
        plot_metrics(metrics)
    plot_grade_distribution(df)
    print(f"\n   All plots saved to '{PLOT_DIR}/' folder.\n")

# ── Standalone run ───────────────────────────────────────────
if __name__ == "__main__":
    generate_all()
