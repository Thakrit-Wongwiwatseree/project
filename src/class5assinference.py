import os
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, roc_auc_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG (MATCH YOUR TRAINING)
# -----------------------------
MODEL_PATH = "examples/model_fixed.h5"
ARTIFACT_DIR = "examples/artifacts_fixed"

DATA_PATH = "data/data.csv"     # change to unseen inference file if needed
TARGET_COL = "ProdTaken"
SEP = ","

THRESHOLD = 0.50

OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1) Load model + artifacts
# -----------------------------
print("Loading trained model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

print("\nLoading artifacts...")
train_columns = joblib.load(os.path.join(ARTIFACT_DIR, "train_columns.pkl"))
scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
print("Artifacts loaded successfully!")

# -----------------------------
# 2) Load data
# -----------------------------
print("\nLoading inference data...")
df = pd.read_csv(DATA_PATH, sep=SEP)
print(f"Dataset shape: {df.shape}")

has_target = TARGET_COL in df.columns

if has_target:
    print(f"\nTarget distribution:\n{df[TARGET_COL].value_counts()}")
    X = df.drop(TARGET_COL, axis=1)
    y_true = df[TARGET_COL].astype(int).to_numpy()
else:
    X = df.copy()
    y_true = None

# -----------------------------
# 3) Preprocess (must match training)
# -----------------------------
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Align to training columns
X_encoded = X_encoded.reindex(columns=train_columns, fill_value=0)

# Scale using training scaler (NO fit)
X_scaled = scaler.transform(X_encoded)

# -----------------------------
# 4) Predict
# -----------------------------
print("\nMaking predictions...")
y_pred_proba = model.predict(X_scaled, verbose=0).ravel()
y_pred = (y_pred_proba > THRESHOLD).astype(int)

# -----------------------------
# 5) Evaluate (ADD F1) only if target exists
# -----------------------------
if has_target:
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS (ANN)")
    print("=" * 60)
    print(f"Threshold: {THRESHOLD}")

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy:  {accuracy:.4f}")

    try:
        auc_score = roc_auc_score(y_true, y_pred_proba)
        print(f"AUC:       {auc_score:.4f}")
    except Exception:
        auc_score = None
        print("AUC:       Could not calculate")

    # F1
    f1_class1 = f1_score(y_true, y_pred, pos_label=1)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(f"F1 (class 1): {f1_class1:.4f}")
    print(f"F1 (macro):   {f1_macro:.4f}")
    print(f"F1 (weighted):{f1_weighted:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["0", "1"]))

    cm = confusion_matrix(y_true, y_pred)

    # Confusion Matrix plot (counts)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
        cbar_kws={"label": "Count"}
    )
    plt.title("Confusion Matrix - Inference (ANN)", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "confusion_matrix_inference.jpg")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # Normalized confusion matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2%", cmap="Greens",
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
        cbar_kws={"label": "Percentage"}
    )
    plt.title("Normalized Confusion Matrix - Inference (ANN)", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    output_path_norm = os.path.join(OUTPUT_DIR, "confusion_matrix_normalized_inference.jpg")
    plt.savefig(output_path_norm, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path_norm}")

    # Save inference metrics for report
    metrics_path = os.path.join(OUTPUT_DIR, "inference_eval_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("ANN MODEL - INFERENCE EVAL\n")
        f.write(f"Threshold: {THRESHOLD}\n\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
        if auc_score is not None:
            f.write(f"AUC: {auc_score:.6f}\n")
        f.write(f"F1_class1: {f1_class1:.6f}\n")
        f.write(f"F1_macro: {f1_macro:.6f}\n")
        f.write(f"F1_weighted: {f1_weighted:.6f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=["0", "1"]))
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print(f"Saved metrics to: {metrics_path}")

# -----------------------------
# 6) Save predictions CSV (always)
# -----------------------------
results_df = df.copy()
results_df["prediction_probability"] = y_pred_proba
results_df["predicted_class"] = y_pred

if has_target:
    results_df["correct_prediction"] = (
        results_df[TARGET_COL].astype(int) == results_df["predicted_class"]
    )

output_csv_path = os.path.join(OUTPUT_DIR, "predictions_inference.csv")
results_df.to_csv(output_csv_path, index=False)
print(f"\nPredictions saved to: {output_csv_path}")

print("\n" + "=" * 60)
print("Inference completed successfully!")
print("=" * 60)

