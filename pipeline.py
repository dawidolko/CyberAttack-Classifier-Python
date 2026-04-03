"""
Cyber Security Attacks Classifier — ML Pipeline
=================================================
Preprocessing, training (Random Forest), evaluation, and artifact export.
All results are saved to results/ for the Streamlit dashboard.
"""

import os
import json
import warnings
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.feature_selection import mutual_info_classif
import joblib

from download_data import ensure_dataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cybersecurity_attacks.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def progress(step: int, total: int, msg: str):
    pct = step / total * 100
    bar_len = 40
    filled = int(bar_len * step // total)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:5.1f}%  {msg:<55s}", end="", flush=True)
    if step == total:
        print()


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    print("\n>> Loading dataset...")
    ensure_dataset(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print(f"   Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. EDA & Save Plots
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    print("\n>> Running Exploratory Data Analysis...")
    total_steps = 6

    # --- class distribution ---
    progress(1, total_steps, "Class distribution plot")
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df["Attack Type"].value_counts()
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    counts.plot.bar(ax=ax, color=colors, edgecolor="black")
    ax.set_title("Attack Type Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Attack Type")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v:,}\n({v / len(df) * 100:.1f}%)", ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)

    # --- missing values ---
    progress(2, total_steps, "Missing values plot")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    missing.plot.barh(ax=ax, color="#e67e22", edgecolor="black")
    ax.set_title("Missing Values per Column", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Missing Values")
    for i, v in enumerate(missing.values):
        ax.text(v + 100, i, f"{v:,} ({v / len(df) * 100:.1f}%)", va="center")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "missing_values.png"), dpi=150)
    plt.close(fig)

    # --- numeric distributions ---
    progress(3, total_steps, "Numeric feature distributions")
    num_cols = ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, num_cols):
        df[col].hist(bins=50, ax=ax, color="#3498db", edgecolor="black", alpha=0.7)
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency")
    plt.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "numeric_distributions.png"), dpi=150)
    plt.close(fig)

    # --- categorical distributions ---
    progress(4, total_steps, "Categorical feature distributions")
    cat_cols = [
        "Protocol", "Packet Type", "Traffic Type", "Severity Level",
        "Action Taken", "Network Segment", "Log Source", "Attack Signature",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, col in zip(axes.flat, cat_cols):
        df[col].value_counts().plot.bar(ax=ax, color="#9b59b6", edgecolor="black")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "categorical_distributions.png"), dpi=150)
    plt.close(fig)

    # --- correlation heatmap (numeric) ---
    progress(5, total_steps, "Correlation heatmap")
    num_df = df[num_cols].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num_df.corr(), annot=True, fmt=".3f", cmap="coolwarm", ax=ax,
                vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    # --- Mutual Information analysis ---
    progress(6, total_steps, "Mutual Information analysis")
    target_le = LabelEncoder()
    y_temp = target_le.fit_transform(df["Attack Type"])

    mi_results = {}
    for col in num_cols:
        mi = mutual_info_classif(df[[col]], y_temp, random_state=42)[0]
        mi_results[col] = float(mi)

    for col in ["Protocol", "Packet Type", "Traffic Type", "Attack Signature",
                "Action Taken", "Severity Level", "Network Segment", "Log Source"]:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col])
        mi = mutual_info_classif(encoded.reshape(-1, 1), y_temp,
                                 discrete_features=True, random_state=42)[0]
        mi_results[col] = float(mi)

    for col in ["Malware Indicators", "Alerts/Warnings"]:
        binary = df[col].notna().astype(int)
        mi = mutual_info_classif(binary.values.reshape(-1, 1), y_temp,
                                 discrete_features=True, random_state=42)[0]
        mi_results[col] = float(mi)

    # MI bar chart
    mi_series = pd.Series(mi_results).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    mi_series.plot.barh(ax=ax, color="#e74c3c", edgecolor="black")
    ax.set_title("Mutual Information with Attack Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mutual Information Score")
    ax.axvline(x=0.01, color="gray", linestyle="--", alpha=0.7, label="Threshold (0.01)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "mutual_information.png"), dpi=150)
    plt.close(fig)

    # Save EDA summary JSON
    eda_summary = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "class_distribution": counts.to_dict(),
        "missing_values": {k: int(v) for k, v in df.isnull().sum().items() if v > 0},
        "numeric_stats": df[num_cols].describe().to_dict(),
        "mutual_information": mi_results,
    }
    with open(os.path.join(RESULTS_DIR, "eda_summary.json"), "w") as f:
        json.dump(eda_summary, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    print("\n>> Preprocessing data...")
    total_steps = 6

    # --- Drop columns not useful for classification ---
    progress(1, total_steps, "Dropping non-predictive columns")
    drop_cols = [
        "Timestamp",
        "Source IP Address",
        "Destination IP Address",
        "Payload Data",
        "User Information",
        "Device Information",
        "Geo-location Data",
        "Proxy Information",
        "Firewall Logs",
        "IDS/IPS Alerts",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # --- Handle missing values ---
    progress(2, total_steps, "Handling missing values")
    df["Malware Indicators"] = df["Malware Indicators"].notna().astype(int)
    df["Alerts/Warnings"] = df["Alerts/Warnings"].notna().astype(int)

    # --- Encode target ---
    progress(3, total_steps, "Encoding target variable")
    target_le = LabelEncoder()
    y = target_le.fit_transform(df["Attack Type"])
    df = df.drop(columns=["Attack Type"])

    # --- Encode categorical features ---
    progress(4, total_steps, "Encoding categorical features")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # --- Scale numeric features ---
    progress(5, total_steps, "Scaling numeric features")
    num_cols = ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores"]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # --- Train/test split ---
    progress(6, total_steps, "Splitting train/test (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Save preprocessing info
    preprocess_info = {
        "dropped_columns": drop_cols,
        "encoded_categorical": cat_cols,
        "scaled_numeric": num_cols,
        "feature_names": df.columns.tolist(),
        "target_classes": target_le.classes_.tolist(),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }
    with open(os.path.join(RESULTS_DIR, "preprocessing_info.json"), "w") as f:
        json.dump(preprocess_info, f, indent=2)

    print(f"   Features: {df.shape[1]} | Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    return X_train, X_test, y_train, y_test, target_le, scaler, label_encoders


# ---------------------------------------------------------------------------
# 4. Train Models
# ---------------------------------------------------------------------------
def train_models(X_train, X_test, y_train, y_test, target_le):
    print("\n>> Training models...")

    class_names = target_le.classes_.tolist()
    all_model_results = {}

    # ===== Random Forest (primary) =====
    print("\n   --- Random Forest ---")
    total_steps = 3
    progress(1, total_steps, "Fitting Random Forest (n_estimators=200)")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    progress(2, total_steps, "Running 5-fold cross-validation")
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)

    progress(3, total_steps, "Saving model to disk")
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.joblib"))

    cv_info = {
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }
    with open(os.path.join(RESULTS_DIR, "cv_scores.json"), "w") as f:
        json.dump(cv_info, f, indent=2)

    print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ===== Gradient Boosting =====
    print("\n   --- Gradient Boosting ---")
    progress(1, 2, "Fitting Gradient Boosting (n_estimators=100)")
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
    )
    gb.fit(X_train, y_train)
    progress(2, 2, "Cross-validation")
    gb_cv = cross_val_score(gb, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"   CV Accuracy: {gb_cv.mean():.4f} (+/- {gb_cv.std():.4f})")

    # ===== k-NN =====
    print("\n   --- k-Nearest Neighbors ---")
    progress(1, 2, "Fitting k-NN (k=7)")
    knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    knn.fit(X_train, y_train)
    progress(2, 2, "Cross-validation")
    knn_cv = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"   CV Accuracy: {knn_cv.mean():.4f} (+/- {knn_cv.std():.4f})")

    # Evaluate all on test set
    models = {"Random Forest": rf, "Gradient Boosting": gb, "k-NN": knn}
    cv_results = {"Random Forest": cv_info, "Gradient Boosting": {
        "cv_scores": gb_cv.tolist(), "cv_mean": float(gb_cv.mean()), "cv_std": float(gb_cv.std()),
    }, "k-NN": {
        "cv_scores": knn_cv.tolist(), "cv_mean": float(knn_cv.mean()), "cv_std": float(knn_cv.std()),
    }}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        roc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

        all_model_results[name] = {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "roc_auc_ovr_macro": float(roc),
            "cv_mean": cv_results[name]["cv_mean"],
            "cv_std": cv_results[name]["cv_std"],
        }

    with open(os.path.join(RESULTS_DIR, "model_comparison.json"), "w") as f:
        json.dump(all_model_results, f, indent=2)

    # --- Model comparison bar chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = list(all_model_results.keys())
    metric_names = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovr_macro"]
    metric_labels = ["Accuracy", "F1 (macro)", "Precision", "Recall", "ROC AUC"]
    x = np.arange(len(metric_labels))
    width = 0.25
    colors_bar = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, (mname, color) in enumerate(zip(model_names, colors_bar)):
        vals = [all_model_results[mname][m] for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=mname, color=color, edgecolor="black")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Test Set)", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1 / 3, color="gray", linestyle="--", alpha=0.7, label="Random baseline (0.333)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)

    return rf, all_model_results


# ---------------------------------------------------------------------------
# 5. Evaluate (detailed — Random Forest)
# ---------------------------------------------------------------------------
def evaluate(rf, X_test, y_test, target_le):
    print("\n>> Detailed evaluation (Random Forest)...")
    total_steps = 4

    class_names = target_le.classes_.tolist()

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    # Metrics
    progress(1, total_steps, "Computing metrics")
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="macro")
    recall_val = recall_score(y_test, y_pred, average="macro")
    roc_auc_ovr = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=class_names)

    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision),
        "recall_macro": float(recall_val),
        "roc_auc_ovr_macro": float(roc_auc_ovr),
        "classification_report": report,
        "classification_report_text": report_text,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n   Accuracy:   {acc:.4f}")
    print(f"   F1 (macro): {f1_macro:.4f}")
    print(f"   ROC AUC:    {roc_auc_ovr:.4f}")
    print(f"\n{report_text}")

    # --- Confusion Matrix ---
    progress(2, total_steps, "Plotting confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=axes[0], linewidths=0.5)
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens", xticklabels=class_names,
                yticklabels=class_names, ax=axes[1], linewidths=0.5)
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    np.save(os.path.join(RESULTS_DIR, "confusion_matrix.npy"), cm)

    # --- ROC Curves ---
    progress(3, total_steps, "Plotting ROC curves")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_roc = ["#e74c3c", "#3498db", "#2ecc71"]
    roc_data = {}
    for i, (cls, color) in enumerate(zip(class_names, colors_roc)):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
        roc_auc_i = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc_i:.3f})")
        roc_data[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc_i)}

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "roc_data.json"), "w") as f:
        json.dump(roc_data, f, indent=2)

    # --- Feature Importance ---
    progress(4, total_steps, "Computing feature importances")
    feat_imp = pd.Series(rf.feature_importances_, index=X_test.columns).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    feat_imp.plot.barh(ax=ax, color="#1abc9c", edgecolor="black")
    ax.set_title("Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)

    feat_imp_dict = feat_imp.sort_values(ascending=False).to_dict()
    with open(os.path.join(RESULTS_DIR, "feature_importance.json"), "w") as f:
        json.dump(feat_imp_dict, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Cyber Security Attacks Classifier")
    print("  ML Pipeline (Random Forest + Comparison)")
    print("=" * 60)

    df = load_data()
    run_eda(df)
    X_train, X_test, y_train, y_test, target_le, scaler, label_encoders = preprocess(df)
    rf, comparison = train_models(X_train, X_test, y_train, y_test, target_le)
    metrics = evaluate(rf, X_test, y_test, target_le)

    # Save all artifacts info
    artifacts = {
        "model_path": os.path.join(MODEL_DIR, "random_forest.joblib"),
        "results_dir": RESULTS_DIR,
        "plots": [
            "class_distribution.png",
            "missing_values.png",
            "numeric_distributions.png",
            "categorical_distributions.png",
            "correlation_heatmap.png",
            "mutual_information.png",
            "model_comparison.png",
            "confusion_matrix.png",
            "roc_curves.png",
            "feature_importance.png",
        ],
    }
    with open(os.path.join(RESULTS_DIR, "artifacts.json"), "w") as f:
        json.dump(artifacts, f, indent=2)

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Model saved to:   {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
