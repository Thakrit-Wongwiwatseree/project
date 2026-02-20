import os
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/data.csv"
TARGET_COL = "ProdTaken"
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_PATH = "examples/model_fixed.h5"
ARTIFACT_DIR = "examples/artifacts_fixed"
OUTPUT_DIR = "data/output"

THRESHOLD = 0.50

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. Load Data
# -----------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Target distribution:\n", df[TARGET_COL].value_counts())

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL].astype(int)

# -----------------------------
# 2. Train/Test Split
# -----------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# -----------------------------
# 3. One-hot encoding
# -----------------------------
cat_cols = X_train_raw.select_dtypes(include=["object"]).columns.tolist()

X_train_enc = pd.get_dummies(X_train_raw, columns=cat_cols, drop_first=True)
X_test_enc = pd.get_dummies(X_test_raw, columns=cat_cols, drop_first=True)

X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

# -----------------------------
# 4. Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled = scaler.transform(X_test_enc)

# -----------------------------
# 5. Build Model
# -----------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(32, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

model.summary()

# -----------------------------
# 6. Train
# -----------------------------
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# -----------------------------
# 7. Evaluate (ADD F1)
# -----------------------------
y_proba = model.predict(X_test_scaled).ravel()
y_pred = (y_proba > THRESHOLD).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# F1 scores
f1_class1 = f1_score(y_test, y_pred, pos_label=1)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print("\n" + "="*60)
print("TRAINING EVALUATION (on test split)")
print("="*60)
print(f"Threshold: {THRESHOLD}")
print(f"Accuracy : {acc:.4f}")
print(f"AUC      : {auc:.4f}")
print(f"F1 (class 1) : {f1_class1:.4f}")
print(f"F1 (macro)   : {f1_macro:.4f}")
print(f"F1 (weighted): {f1_weighted:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["0", "1"]))

cm = confusion_matrix(y_test, y_pred)

# Save metrics for report
metrics_path = os.path.join(OUTPUT_DIR, "training_eval_metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("ANN MODEL - TRAINING EVAL (TEST SPLIT)\n")
    f.write(f"Threshold: {THRESHOLD}\n\n")
    f.write(f"Accuracy: {acc:.6f}\n")
    f.write(f"AUC: {auc:.6f}\n")
    f.write(f"F1_class1: {f1_class1:.6f}\n")
    f.write(f"F1_macro: {f1_macro:.6f}\n")
    f.write(f"F1_weighted: {f1_weighted:.6f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=["0", "1"]))
    f.write("\n\nConfusion Matrix:\n")
    f.write(np.array2string(cm))
print(f"Saved metrics to: {metrics_path}")

# -----------------------------
# 8. Confusion Matrix Plot (ONLY PLOTS; NO LINE GRAPHS)
# -----------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["0", "1"],
            yticklabels=["0", "1"])
plt.title("Confusion Matrix - Training Eval (ANN)")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_train_eval.jpg"), dpi=300)
plt.close()
print("Saved: confusion_matrix_train_eval.jpg")

# Normalized confusion matrix
plt.figure(figsize=(8, 6))
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Greens",
            xticklabels=["0", "1"],
            yticklabels=["0", "1"])
plt.title("Normalized Confusion Matrix - Training Eval (ANN)")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_train_eval_normalized.jpg"), dpi=300)
plt.close()
print("Saved: confusion_matrix_train_eval_normalized.jpg")

# -----------------------------
# 9. Save Model + Artifacts
# -----------------------------
model.save(MODEL_PATH)
joblib.dump(X_train_enc.columns.tolist(), os.path.join(ARTIFACT_DIR, "train_columns.pkl"))
joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))

print("Model and artifacts saved successfully.")

