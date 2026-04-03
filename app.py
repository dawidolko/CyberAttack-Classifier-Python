"""
Cyber Security Attacks Classifier — Streamlit Dashboard
========================================================
Interactive dashboard showing EDA, preprocessing, model results, and analysis.
"""

import os
import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from download_data import ensure_dataset

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cyber Attack Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cybersecurity_attacks.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_csv():
    ensure_dataset(DATA_PATH)
    return pd.read_csv(DATA_PATH)


def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_image(name):
    path = os.path.join(RESULTS_DIR, name)
    return path if os.path.exists(path) else None


def check_results():
    return os.path.exists(os.path.join(RESULTS_DIR, "metrics.json"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ Cyber Attack Classifier")
st.sidebar.markdown("---")

if not check_results():
    st.sidebar.warning("⚠️ Pipeline results not found. Run `python pipeline.py` first.")

tab_names = [
    "📋 Project Overview",
    "📊 Exploratory Data Analysis",
    "⚙️ Preprocessing",
    "🌲 Model & Training",
    "⚖️ Model Comparison",
    "📈 Results & Evaluation",
    "🔍 Interactive Explorer",
]
selected_tab = st.sidebar.radio("Navigation", tab_names)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** Cyber Security Attacks (Kaggle)  \n"
    "**Model:** Random Forest Classifier  \n"
    "**Classes:** DDoS, Malware, Intrusion"
)

# ===================================================================
# TAB 1: Project Overview
# ===================================================================
if selected_tab == tab_names[0]:
    st.title("🛡️ Cyber Security Attacks Classifier")
    st.markdown("### Multiclass Classification with Random Forest")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Instances", "40,000")
    col2.metric("Attributes", "25")
    col3.metric("Classes", "3")

    st.markdown("#### Problem Description")
    st.markdown(
        """
        This project performs **multiclass classification** of network attacks using the
        [Cyber Security Attacks Dataset](https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks)
        from Kaggle. The goal is to classify network traffic into one of three attack types:

        | Attack Type | Description | Distribution |
        |------------|-------------|-------------|
        | **DDoS** | Distributed Denial of Service | ~33.6% |
        | **Malware** | Malicious Software | ~33.3% |
        | **Intrusion** | Network Intrusion | ~33.2% |

        The dataset contains 40,000 records with 25 attributes describing network packets,
        traffic characteristics, malware indicators, anomaly scores, security alerts, and
        defensive actions taken.
        """
    )

    st.markdown("#### Attributes Overview")
    attr_data = {
        "Attribute": [
            "Timestamp", "Source IP Address", "Destination IP Address",
            "Source Port", "Destination Port", "Protocol", "Packet Length",
            "Packet Type", "Traffic Type", "Payload Data", "Malware Indicators",
            "Anomaly Scores", "Alerts/Warnings", "Attack Type", "Attack Signature",
            "Action Taken", "Severity Level", "User Information", "Device Information",
            "Network Segment", "Geo-location Data", "Proxy Information",
            "Firewall Logs", "IDS/IPS Alerts", "Log Source"
        ],
        "Type": [
            "Datetime", "Categorical (IP)", "Categorical (IP)",
            "Numeric", "Numeric", "Categorical", "Numeric",
            "Categorical", "Categorical", "Text", "Categorical (~50% missing)",
            "Numeric", "Categorical (~50% missing)", "Target", "Categorical",
            "Categorical", "Categorical", "Categorical", "Text (User Agent)",
            "Categorical", "Categorical", "Categorical (~50% missing)",
            "Categorical (~50% missing)", "Categorical (~50% missing)", "Categorical"
        ],
        "Description": [
            "Event date and time", "Source IP address", "Destination IP address",
            "Source port number (1027-65530)", "Destination port number (1024-65535)",
            "Network protocol (TCP/UDP/ICMP)", "Packet size in bytes (64-1500)",
            "Packet type (Data/Control)", "Traffic type (HTTP/DNS/FTP)",
            "Packet payload content (text)", "Malware indicator presence",
            "Anomaly score (0-100, continuous)", "System alert presence",
            "DDoS / Malware / Intrusion", "Known Pattern A / Known Pattern B",
            "Logged / Blocked / Ignored", "Low / Medium / High",
            "User identifier", "User agent string",
            "Network segment (A/B/C)", "Geographic location",
            "Proxy IP address", "Firewall log data",
            "IDS/IPS alert data", "Log source (Server/Firewall)"
        ],
    }
    st.dataframe(pd.DataFrame(attr_data), width="stretch", hide_index=True)

    st.markdown("#### Model")
    st.markdown(
        """
        **Random Forest Classifier** with 200 estimators, max_depth=20, balanced class weights.
        Evaluated using 5-fold stratified cross-validation and a held-out 20% test set.
        """
    )

# ===================================================================
# TAB 2: EDA
# ===================================================================
elif selected_tab == tab_names[1]:
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    df = load_csv()
    eda_summary = load_json("eda_summary.json")

    # Dataset shape
    st.markdown("### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    missing_total = df.isnull().sum().sum()
    c3.metric("Missing Values", f"{missing_total:,}")
    c4.metric("Missing %", f"{missing_total / (df.shape[0] * df.shape[1]) * 100:.1f}%")

    # Sample data
    st.markdown("### Sample Data (first 10 rows)")
    st.dataframe(df.head(10), width="stretch")

    # Class distribution
    st.markdown("### Target Variable: Attack Type")
    col1, col2 = st.columns([2, 1])
    with col1:
        counts = df["Attack Type"].value_counts()
        fig = px.bar(
            x=counts.index, y=counts.values,
            color=counts.index,
            color_discrete_map={"DDoS": "#e74c3c", "Malware": "#3498db", "Intrusion": "#2ecc71"},
            labels={"x": "Attack Type", "y": "Count"},
            title="Attack Type Distribution",
            text=counts.values,
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index,
            color_discrete_map={"DDoS": "#e74c3c", "Malware": "#3498db", "Intrusion": "#2ecc71"},
            title="Class Balance",
        )
        st.plotly_chart(fig, width="stretch")

    # Missing values
    st.markdown("### Missing Values Analysis")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        fig = px.bar(
            x=missing.values, y=missing.index,
            orientation="h",
            labels={"x": "Missing Count", "y": "Column"},
            title="Columns with Missing Values",
            text=[f"{v:,} ({v/len(df)*100:.1f}%)" for v in missing.values],
            color_discrete_sequence=["#e67e22"],
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")
    else:
        st.success("No missing values found!")

    # Numeric distributions
    st.markdown("### Numeric Feature Distributions")
    num_cols = ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores"]
    num_tab1, num_tab2 = st.tabs(["Histograms", "Box Plots"])
    with num_tab1:
        fig = make_subplots(rows=2, cols=2, subplot_titles=num_cols)
        for i, col in enumerate(num_cols):
            r, c = divmod(i, 2)
            fig.add_trace(
                go.Histogram(x=df[col], nbinsx=50, marker_color="#3498db", name=col),
                row=r + 1, col=c + 1,
            )
        fig.update_layout(height=600, showlegend=False, title_text="Numeric Feature Histograms")
        st.plotly_chart(fig, width="stretch")
    with num_tab2:
        fig = make_subplots(rows=1, cols=4, subplot_titles=num_cols)
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
        for i, (col, color) in enumerate(zip(num_cols, colors)):
            fig.add_trace(
                go.Box(y=df[col], name=col, marker_color=color),
                row=1, col=i + 1,
            )
        fig.update_layout(height=400, showlegend=False, title_text="Numeric Feature Box Plots")
        st.plotly_chart(fig, width="stretch")

    # Categorical distributions
    st.markdown("### Categorical Feature Distributions")
    cat_cols = ["Protocol", "Packet Type", "Traffic Type", "Severity Level",
                "Action Taken", "Network Segment", "Log Source", "Attack Signature"]
    fig = make_subplots(rows=2, cols=4, subplot_titles=cat_cols)
    cat_colors = px.colors.qualitative.Set2
    for i, col in enumerate(cat_cols):
        r, c = divmod(i, 4)
        vc = df[col].value_counts()
        fig.add_trace(
            go.Bar(x=vc.index, y=vc.values, marker_color=cat_colors[i], name=col),
            row=r + 1, col=c + 1,
        )
    fig.update_layout(height=600, showlegend=False, title_text="Categorical Feature Distributions")
    st.plotly_chart(fig, width="stretch")

    # Correlation heatmap
    st.markdown("### Correlation Heatmap (Numeric Features)")
    corr = df[num_cols].corr()
    fig = px.imshow(
        corr, text_auto=".3f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Feature Correlation Matrix",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

# ===================================================================
# TAB 3: Preprocessing
# ===================================================================
elif selected_tab == tab_names[2]:
    st.title("⚙️ Data Preprocessing")
    st.markdown("---")

    preprocess_info = load_json("preprocessing_info.json")

    st.markdown("### Preprocessing Pipeline")
    st.markdown(
        """
        The following preprocessing steps were applied to prepare the data for the Random Forest model:
        """
    )

    steps = [
        ("1. Drop Non-Predictive Columns",
         "Columns that cannot generalize for classification (IP addresses, timestamps, free-text fields, "
         "columns with ~50% missing and only one unique value) were removed.",
         ["Timestamp", "Source IP Address", "Destination IP Address", "Payload Data",
          "User Information", "Device Information", "Geo-location Data",
          "Proxy Information", "Firewall Logs", "IDS/IPS Alerts"]),
        ("2. Handle Missing Values",
         "Binary columns (Malware Indicators, Alerts/Warnings) had ~50% missing values. "
         "Since present values only had one unique value ('IoC Detected' / 'Alert Triggered'), "
         "they were converted to binary flags (0 = absent, 1 = present).",
         None),
        ("3. Encode Target Variable",
         "The target variable 'Attack Type' was label-encoded: DDoS=0, Intrusion=1, Malware=2.",
         None),
        ("4. Encode Categorical Features",
         "Remaining categorical features were label-encoded for the tree-based model.",
         None),
        ("5. Scale Numeric Features",
         "Numeric features were standardized (zero mean, unit variance) using StandardScaler.",
         ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores"]),
        ("6. Train/Test Split",
         "Data was split 80/20 with stratification to preserve class balance.",
         None),
    ]

    for title, desc, details in steps:
        with st.expander(title, expanded=True):
            st.markdown(desc)
            if details:
                st.code(", ".join(details))

    if preprocess_info:
        st.markdown("### Preprocessing Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Features Used", len(preprocess_info["feature_names"]))
        col2.metric("Training Set", f"{preprocess_info['train_size']:,}")
        col3.metric("Test Set", f"{preprocess_info['test_size']:,}")

        st.markdown("#### Final Feature Set")
        feat_df = pd.DataFrame({
            "Feature": preprocess_info["feature_names"],
            "Index": range(len(preprocess_info["feature_names"])),
        })
        st.dataframe(feat_df, width="stretch", hide_index=True)

# ===================================================================
# TAB 4: Model & Training
# ===================================================================
elif selected_tab == tab_names[3]:
    st.title("🌲 Model Design & Training")
    st.markdown("---")

    st.markdown("### Random Forest Classifier")
    st.markdown(
        """
        **Random Forest** is an ensemble learning method that constructs multiple decision trees
        during training and outputs the class that is the mode of the classes of the individual trees.

        **Why Random Forest?**
        - Handles both numerical and categorical features well
        - Robust to overfitting due to ensemble averaging
        - Provides feature importance rankings
        - Works well with balanced and imbalanced datasets
        - No need for extensive hyperparameter tuning
        """
    )

    st.markdown("### Hyperparameters")
    params = {
        "Parameter": [
            "n_estimators", "max_depth", "min_samples_split",
            "min_samples_leaf", "class_weight", "random_state", "n_jobs"
        ],
        "Value": ["200", "20", "5", "2", "balanced", "42", "-1 (all cores)"],
        "Description": [
            "Number of trees in the forest",
            "Maximum depth of each tree",
            "Minimum samples required to split a node",
            "Minimum samples required at a leaf node",
            "Automatic class weight balancing",
            "Seed for reproducibility",
            "Parallel processing across all CPU cores",
        ],
    }
    st.dataframe(pd.DataFrame(params), width="stretch", hide_index=True)

    st.markdown("### Cross-Validation Results")
    cv_data = load_json("cv_scores.json")
    if cv_data:
        col1, col2 = st.columns(2)
        col1.metric("CV Accuracy (mean)", f"{cv_data['cv_mean']:.4f}")
        col2.metric("CV Accuracy (std)", f"± {cv_data['cv_std']:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=cv_data["cv_scores"],
            marker_color=["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"],
            text=[f"{s:.4f}" for s in cv_data["cv_scores"]],
            textposition="outside",
        ))
        fig.add_hline(y=cv_data["cv_mean"], line_dash="dash", line_color="red",
                      annotation_text=f"Mean = {cv_data['cv_mean']:.4f}")
        fig.update_layout(
            title="5-Fold Cross-Validation Accuracy",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1.05],
            height=400,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Run the pipeline first to see cross-validation results.")

    st.markdown("### How Random Forest Works")
    st.markdown(
        """
        ```
        Training Data
             │
             ├── Bootstrap Sample 1 ──→ Decision Tree 1 ──→ Prediction 1
             ├── Bootstrap Sample 2 ──→ Decision Tree 2 ──→ Prediction 2
             ├── Bootstrap Sample 3 ──→ Decision Tree 3 ──→ Prediction 3
             │   ...                     ...                  ...
             └── Bootstrap Sample N ──→ Decision Tree N ──→ Prediction N
                                                                │
                                                    Majority Vote (Mode)
                                                                │
                                                        Final Prediction
        ```

        Each tree is trained on a random subset of the data (with replacement) and considers
        a random subset of features at each split. This randomness reduces correlation between
        trees and leads to a more robust ensemble.
        """
    )

# ===================================================================
# TAB 5: Model Comparison (RF vs GB vs k-NN)
# ===================================================================
elif selected_tab == tab_names[4]:
    st.title("⚖️ Model Comparison")
    st.markdown("---")

    st.markdown(
        """
        Alongside the primary **Random Forest** classifier, we train **Gradient Boosting** and
        **k-Nearest Neighbors (k=7)** on the same preprocessed train/test split. Metrics below are
        computed on the **held-out test set** (20%). Cross-validation scores use 5-fold
        stratified splits on the training portion only.
        """
    )

    comp = load_json("model_comparison.json")
    if not comp:
        st.warning("⚠️ Run `python pipeline.py` (or `./start.sh`) to generate model comparison results.")
        st.stop()

    rows = []
    for name, m in comp.items():
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "F1 (macro)": m["f1_macro"],
            "Precision (macro)": m["precision_macro"],
            "Recall (macro)": m["recall_macro"],
            "ROC AUC (OvR)": m["roc_auc_ovr_macro"],
            "CV mean": m["cv_mean"],
            "CV std": m["cv_std"],
        })
    comp_df = pd.DataFrame(rows)
    st.dataframe(
        comp_df.style.format({
            "Accuracy": "{:.4f}",
            "F1 (macro)": "{:.4f}",
            "Precision (macro)": "{:.4f}",
            "Recall (macro)": "{:.4f}",
            "ROC AUC (OvR)": "{:.4f}",
            "CV mean": "{:.4f}",
            "CV std": "{:.4f}",
        }),
        width="stretch",
        hide_index=True,
    )

    melt = comp_df.melt(id_vars=["Model"], value_vars=[
        "Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)", "ROC AUC (OvR)",
    ], var_name="Metric", value_name="Score")
    fig = px.bar(
        melt, x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_map={
            "Random Forest": "#e74c3c",
            "Gradient Boosting": "#3498db",
            "k-NN": "#2ecc71",
        },
        title="Test-set metrics by model",
    )
    fig.update_layout(yaxis_range=[0, 1.05], height=500)
    st.plotly_chart(fig, width="stretch")

    img = load_image("model_comparison.png")
    if img:
        st.markdown("### Static comparison chart (from pipeline)")
        st.image(img, width="stretch")

# ===================================================================
# TAB 6: Results & Evaluation
# ===================================================================
elif selected_tab == tab_names[5]:
    st.title("📈 Results & Evaluation")
    st.markdown("---")

    metrics = load_json("metrics.json")
    roc_data = load_json("roc_data.json")
    feat_imp = load_json("feature_importance.json")

    if not metrics:
        st.warning("⚠️ No results found. Run `python pipeline.py` first.")
        st.stop()

    # Key metrics
    st.markdown("### Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("F1 Score (macro)", f"{metrics['f1_macro']:.4f}")
    c3.metric("Precision (macro)", f"{metrics['precision_macro']:.4f}")
    c4.metric("Recall (macro)", f"{metrics['recall_macro']:.4f}")

    st.metric("ROC AUC (OvR, macro)", f"{metrics['roc_auc_ovr_macro']:.4f}")

    # Classification report
    st.markdown("### Classification Report")
    report = metrics["classification_report"]
    report_df = pd.DataFrame({
        k: v for k, v in report.items()
        if k not in ["accuracy", "macro avg", "weighted avg"]
    }).T
    report_df.index.name = "Class"
    st.dataframe(report_df.style.format("{:.4f}"), width="stretch")

    # Averages
    avg_df = pd.DataFrame({
        k: v for k, v in report.items()
        if k in ["macro avg", "weighted avg"]
    }).T
    avg_df.index.name = "Average"
    st.dataframe(avg_df.style.format("{:.4f}"), width="stretch")

    # Confusion matrix
    st.markdown("### Confusion Matrix")
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.npy")
    if os.path.exists(cm_path):
        cm = np.load(cm_path)
        preprocess_info = load_json("preprocessing_info.json")
        class_names = preprocess_info["target_classes"] if preprocess_info else ["DDoS", "Intrusion", "Malware"]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.imshow(
                cm, text_auto=True,
                x=class_names, y=class_names,
                color_continuous_scale="Blues",
                title="Confusion Matrix (Counts)",
                labels={"x": "Predicted", "y": "True", "color": "Count"},
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, width="stretch")

        with col2:
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fig = px.imshow(
                cm_norm, text_auto=".2%",
                x=class_names, y=class_names,
                color_continuous_scale="Greens",
                title="Confusion Matrix (Normalized)",
                labels={"x": "Predicted", "y": "True", "color": "Ratio"},
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, width="stretch")

    # ROC Curves
    st.markdown("### ROC Curves")
    if roc_data:
        fig = go.Figure()
        colors_roc = {"DDoS": "#e74c3c", "Intrusion": "#3498db", "Malware": "#2ecc71"}
        for cls, data in roc_data.items():
            fig.add_trace(go.Scatter(
                x=data["fpr"], y=data["tpr"],
                name=f"{cls} (AUC = {data['auc']:.3f})",
                line=dict(color=colors_roc.get(cls, "#333"), width=2),
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random",
            line=dict(color="gray", width=1, dash="dash"),
        ))
        fig.update_layout(
            title="ROC Curves (One-vs-Rest)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500,
        )
        st.plotly_chart(fig, width="stretch")

    # Feature importance
    st.markdown("### Feature Importances")
    if feat_imp:
        fig = px.bar(
            x=list(feat_imp.values()), y=list(feat_imp.keys()),
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            title="Random Forest Feature Importances",
            color=list(feat_imp.values()),
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

# ===================================================================
# TAB 7: Interactive Explorer
# ===================================================================
elif selected_tab == tab_names[6]:
    st.title("🔍 Interactive Data Explorer")
    st.markdown("---")

    df = load_csv()

    st.markdown("### Filter & Explore the Dataset")

    col1, col2, col3 = st.columns(3)
    with col1:
        attack_filter = st.multiselect(
            "Attack Type", df["Attack Type"].unique().tolist(),
            default=df["Attack Type"].unique().tolist()
        )
    with col2:
        protocol_filter = st.multiselect(
            "Protocol", df["Protocol"].unique().tolist(),
            default=df["Protocol"].unique().tolist()
        )
    with col3:
        severity_filter = st.multiselect(
            "Severity Level", df["Severity Level"].unique().tolist(),
            default=df["Severity Level"].unique().tolist()
        )

    filtered = df[
        (df["Attack Type"].isin(attack_filter)) &
        (df["Protocol"].isin(protocol_filter)) &
        (df["Severity Level"].isin(severity_filter))
    ]

    st.markdown(f"**Showing {len(filtered):,} / {len(df):,} records**")

    # Interactive scatter
    st.markdown("### Feature Scatter Plot")
    num_cols = ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores"]
    sc1, sc2 = st.columns(2)
    with sc1:
        x_feat = st.selectbox("X-axis", num_cols, index=2)
    with sc2:
        y_feat = st.selectbox("Y-axis", num_cols, index=3)

    sample = filtered.sample(min(5000, len(filtered)), random_state=42)
    fig = px.scatter(
        sample, x=x_feat, y=y_feat,
        color="Attack Type",
        color_discrete_map={"DDoS": "#e74c3c", "Malware": "#3498db", "Intrusion": "#2ecc71"},
        opacity=0.5,
        title=f"{x_feat} vs {y_feat} (sampled, max 5000 points)",
        hover_data=["Protocol", "Severity Level", "Traffic Type"],
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

    # Distribution by attack type
    st.markdown("### Distribution by Attack Type")
    dist_col = st.selectbox("Select feature", num_cols + ["Protocol", "Traffic Type", "Severity Level", "Action Taken"])
    if dist_col in num_cols:
        fig = px.histogram(
            filtered, x=dist_col, color="Attack Type",
            color_discrete_map={"DDoS": "#e74c3c", "Malware": "#3498db", "Intrusion": "#2ecc71"},
            barmode="overlay", opacity=0.7, nbins=50,
            title=f"Distribution of {dist_col} by Attack Type",
        )
    else:
        ct = filtered.groupby([dist_col, "Attack Type"]).size().reset_index(name="Count")
        fig = px.bar(
            ct, x=dist_col, y="Count", color="Attack Type",
            color_discrete_map={"DDoS": "#e74c3c", "Malware": "#3498db", "Intrusion": "#2ecc71"},
            barmode="group",
            title=f"Distribution of {dist_col} by Attack Type",
        )
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")

    # Data table
    st.markdown("### Filtered Data Table")
    display_cols = st.multiselect(
        "Columns to display",
        filtered.columns.tolist(),
        default=["Timestamp", "Source IP Address", "Protocol", "Packet Length",
                 "Anomaly Scores", "Attack Type", "Severity Level", "Action Taken"],
    )
    st.dataframe(filtered[display_cols].head(100), width="stretch", hide_index=True)
