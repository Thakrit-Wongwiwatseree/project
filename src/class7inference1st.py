import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG
# -----------------------------
BASELINE_DIR = "examples/baseline_logreg"
MODEL_PATH = os.path.join(BASELINE_DIR, "logreg_model.pkl")
ARTIFACT_DIR = os.path.join(BASELINE_DIR, "artifacts")

DATA_PATH = "data/data.csv"   # replace with unseen file if needed
TARGET_COL = "ProdTaken"
SEP = ","
THRESHOLD = 0.50

OUTPUT_DIR = "data/output_baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1) Load model + artifacts
# -----------------------------
print("Loading baseline model + artifacts...")
model = joblib.load(MODEL_PATH)
train_columns = joblib.load(os.path.join(ARTIFACT_DIR, "train_columns.pkl"))
scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
print("Loaded successfully!")

# -----------------------------
# 2) Load data
# -----------------------------
df = pd.read_csv(DATA_PATH, sep=SEP)
print("Inference dataset shape:", df.shape)

has_target = TARGET_COL in df.columns
if has_target:
    X = df.drop(TARGET_COL, axis=1)
    y_true = df[TARGET_COL].astype(int).to_numpy()
else:
    X = df.copy()
    y_true = None

# -----------------------------
# 3) Preprocess (match training)
# -----------------------------
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X_enc = X_enc.reindex(columns=train_columns, fill_value=0)
X_scaled = scaler.transform(X_enc)

# -----------------------------
# 4) Predict
# -----------------------------
y_pred_proba = model.predict_proba(X_scaled)[:, 1]
y_pred = (y_pred_proba > THRESHOLD).astype(int)

# -----------------------------
# 5) Evaluate (ADD F1)
# -----------------------------
if has_target:
    acc = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except Exception:
        auc = None

    f1_class1 = f1_score(y_true, y_pred, pos_label=1)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print("\nBASELINE INFERENCE RESULTS")
    print(f"Threshold: {THRESHOLD}")
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}" if auc is not None else "AUC: Could not calculate")
    print(f"F1 (class 1): {f1_class1:.4f}")
    print(f"F1 (macro)  : {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["0", "1"]))

    cm = confusion_matrix(y_true, y_pred)

    # Confusion Matrix (counts)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.title("Baseline Confusion Matrix - Inference (LogReg)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "baseline_confusion_matrix_inference.jpg"), dpi=300)
    plt.close()

    # Confusion Matrix (normalized)
    plt.figure(figsize=(8, 6))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Greens",
                xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.title("Baseline Normalized Confusion Matrix - Inference (LogReg)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "baseline_confusion_matrix_norm_inference.jpg"), dpi=300)
    plt.close()

    # Save inference metrics for report too
    metrics_path = os.path.join(OUTPUT_DIR, "baseline_inference_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("BASELINE INFERENCE: Logistic Regression\n")
        f.write(f"Threshold: {THRESHOLD}\n\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        if auc is not None:
            f.write(f"AUC: {auc:.6f}\n")
        f.write(f"F1_class1: {f1_class1:.6f}\n")
        f.write(f"F1_macro: {f1_macro:.6f}\n")
        f.write(f"F1_weighted: {f1_weighted:.6f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=["0", "1"]))
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print("Saved:", metrics_path)

# -----------------------------
# 6) Save predictions CSV
# -----------------------------
results_df = df.copy()
results_df["prediction_probability"] = y_pred_proba
results_df["predicted_class"] = y_pred
if has_target:
    results_df["correct_prediction"] = (results_df[TARGET_COL].astype(int) == results_df["predicted_class"])

out_csv = os.path.join(OUTPUT_DIR, "baseline_predictions_inference.csv")
results_df.to_csv(out_csv, index=False)
print("\nSaved predictions to:", out_csv)
print("Baseline inference completed successfully!")