import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import os

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = r"C:\Coding\ML Project"
model_path = os.path.join(BASE_DIR, "eeg_model.h5")
history_path = os.path.join(BASE_DIR, "history.npy")
X_path = os.path.join(BASE_DIR, "X_segments.npy")
y_path = os.path.join(BASE_DIR, "y_segments.npy")
labels_path = os.path.join(BASE_DIR, "label_classes.npy")

# -------------------------------
# Load files
# -------------------------------
print("📂 Loading data and model...")
model = load_model(model_path)
history = np.load(history_path, allow_pickle=True).item()
X = np.load(X_path, allow_pickle=True)
y = np.load(y_path, allow_pickle=True)
label_classes = np.load(labels_path, allow_pickle=True)

# -------------------------------
# Plot Accuracy & Loss
# -------------------------------
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="Train Accuracy", marker='o')
plt.plot(history["val_accuracy"], label="Validation Accuracy", marker='o')
plt.title("Model Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Train Loss", marker='o')
plt.plot(history["val_loss"], label="Validation Loss", marker='o')
plt.title("Model Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Save accuracy/loss plot
acc_loss_path = os.path.join(BASE_DIR, "training_curves.png")
plt.tight_layout()
plt.savefig(acc_loss_path, dpi=300)
print(f"✅ Saved training curves at: {acc_loss_path}")

plt.close()

# -------------------------------
# Confusion Matrix
# -------------------------------
print("📊 Generating confusion matrix...")
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.array([np.where(label_classes == lbl)[0][0] for lbl in y])
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=label_classes, yticklabels=label_classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Save confusion matrix
cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(cm_path, dpi=300)
print(f"✅ Saved confusion matrix at: {cm_path}")

plt.close()

# -------------------------------
# Classification Report
# -------------------------------
print("\n📋 Classification Report:")
report = classification_report(y_true, y_pred, target_names=label_classes)
print(report)

# Save report as text file
report_path = os.path.join(BASE_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"✅ Saved classification report at: {report_path}")

print("\n🎉 All plots and reports have been generated successfully!")
