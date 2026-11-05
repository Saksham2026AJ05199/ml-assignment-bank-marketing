import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# 1. Page Header (Left-aligned, clean academic layout)
# ---------------------------------------------------------
st.set_page_config(page_title="ML Assignment 2: Classification", layout="wide")

st.title("Machine Learning Assignment 2: Classification Models")
st.caption("Student Project: Bank Marketing Prediction | BITS Pilani WILP - M.Tech (AIML)")
st.divider()

# ---------------------------------------------------------
# 2. Sidebar: Model Selection (Rubric b: 1 Mark)
# ---------------------------------------------------------
st.sidebar.header("1. Model Selection")

models_dict = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

selected_model_name = st.sidebar.selectbox(
    "Choose a model:",
    list(models_dict.keys()),
    index=4  # Default to Support Vector Machine (Overall Winner)
)

# ---------------------------------------------------------
# 3. Sidebar: Dataset Upload Option (Rubric a: 1 Mark)
# ---------------------------------------------------------
st.sidebar.header("2. Dataset Option")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (must contain input features & 'y' target):", 
    type=["csv"]
)

# ---------------------------------------------------------
# 4. Helper Functions: Caching & Data Loading (No Retraining)
# ---------------------------------------------------------
@st.cache_resource
def load_saved_model(filename):
    model_path = os.path.join("model", filename)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    if os.path.exists(filename):
        return joblib.load(filename)
    return None

def get_test_data():
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "Uploaded CSV File"
    else:
        # Check standard locations for saved test file
        possible_paths = [
            os.path.join("model", "test.csv"),
            os.path.join("model", "test_data.csv"),
            "test_data.csv",
            "test.csv"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return pd.read_csv(p), f"Default Test Data (`{p}`)"
        return None, "None"

# Load the selected model and dataset
selected_file = models_dict[selected_model_name]
model = load_saved_model(selected_file)
df_test, data_source_name = get_test_data()

# ---------------------------------------------------------
# 5. Validation & Inference
# ---------------------------------------------------------
if model is None:
    st.error(f"Error: Model file `{selected_file}` could not be found inside the `model/` directory.")
    st.stop()

if df_test is None:
    st.warning("Please upload a test CSV file or place `test.csv` in the `model/` directory.")
    st.stop()

# Locate target column ('y')
target_col = None
for col in ['y', 'target', 'deposit', 'class']:
    if col in df_test.columns:
        target_col = col
        break

if target_col is None:
    st.error("Error: The test dataset must contain the ground-truth target column ('y').")
    st.stop()

# Separate input features and ground truth
X_test = df_test.drop(columns=[target_col])
y_raw = df_test[target_col]

# Convert target to binary (1 for 'yes', 0 for 'no')
if y_raw.dtype == object:
    y_test = (y_raw.astype(str).str.strip().str.lower() == 'yes').astype(int)
else:
    y_test = y_raw.astype(int)

# Run Inference (No Retraining)
try:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
except Exception as e:
    st.error(f"Inference error: {e}")
    st.stop()

# ---------------------------------------------------------
# 6. Evaluation Metrics Display (Rubric c: 1 Mark)
# ---------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

st.subheader(f"Evaluation Metrics: {selected_model_name}")
st.write(f"Data Source: **{data_source_name}** ({df_test.shape[0]} samples)")

# 6 Clean Native Streamlit Metric Cards
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{acc:.4f}")
c2.metric("AUC Score", f"{auc:.4f}")
c3.metric("Precision", f"{prec:.4f}")
c4.metric("Recall", f"{rec:.4f}")
c5.metric("F1 Score", f"{f1:.4f}")
c6.metric("MCC Score", f"{mcc:.4f}")

st.divider()

# ---------------------------------------------------------
# 7. Confusion Matrix & Classification Report (Rubric d: 1 Mark)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Confusion Matrix", 
    "Classification Report", 
    "All Models Comparison"
])

with tab1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    col_plot, col_info = st.columns([1, 1.2])
    with col_plot:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No (0)', 'Yes (1)'],
            yticklabels=['No (0)', 'Yes (1)'],
            ax=ax, annot_kws={"size": 11}
        )
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    with col_info:
        tn, fp, fn, tp = cm.ravel()
        st.write("**Breakdown:**")
        st.write(f"- True Negatives (TN): **{tn}**")
        st.write(f"- False Positives (FP): **{fp}**")
        st.write(f"- False Negatives (FN): **{fn}**")
        st.write(f"- True Positives (TP): **{tp}**")
        st.write(f"Sensitivity / Recall: **{rec * 100:.2f}%**")

with tab2:
    st.subheader("Classification Report")
    rep = classification_report(
        y_test, y_pred, 
        target_names=['No (0)', 'Yes (1)'], 
        output_dict=True, 
        zero_division=0
    )
    rep_df = pd.DataFrame(rep).transpose().round(4)
    st.dataframe(rep_df, use_container_width=True)

with tab3:
    st.subheader("Comparison Table Across All 5 Models")
    comp_path = os.path.join("model", "model_comparison.csv")
    if not os.path.exists(comp_path):
        comp_path = "model_comparison.csv"

    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("`model_comparison.csv` not found in model/ folder.")

st.caption("BITS Pilani WILP M.Tech (AIML) | Machine Learning Assignment 2")
