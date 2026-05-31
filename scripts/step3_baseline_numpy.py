# ================================================
# 62533 Applied Machine Learning and Big Data
# Brain Tumour MRI Classification
# Author: Simin Asadi
# Study Number: s234955
# Date: May 2026
# ================================================

import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# LOAD PREPARED DATA
# -------------------------------------------------
X_train = np.load(MODELS_DIR / "X_train.npy")
y_train = np.load(MODELS_DIR / "y_train.npy")

X_test = np.load(MODELS_DIR / "X_test.npy")
y_test = np.load(MODELS_DIR / "y_test.npy")

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]

print("Loaded data:")
print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)

# -------------------------------------------------
# FLATTEN IMAGES
# -------------------------------------------------
X_train = X_train.reshape(len(X_train), -1)
X_test  = X_test.reshape(len(X_test), -1)

print("\nAfter flattening:")
print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)

# -------------------------------------------------
# TRAIN / VALIDATION SPLIT (80/20)
# -------------------------------------------------
np.random.seed(42)
n_total    = len(X_train)
n_val      = int(0.2 * n_total)          # 20% = 1120 images
indices    = np.random.permutation(n_total)
val_idx    = indices[:n_val]
train_idx  = indices[n_val:]

X_tr  = X_train[train_idx]              # 4480 images for training
y_tr  = y_train[train_idx]
X_val = X_train[val_idx]               # 1120 images for validation
y_val = y_train[val_idx]

print(f"\nTraining samples:   {len(X_tr)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples:       {len(X_test)}")

# -------------------------------------------------
# ONE-HOT ENCODING
# -------------------------------------------------
def one_hot(y, num_classes):
    y_onehot = np.zeros((len(y), num_classes))
    y_onehot[np.arange(len(y)), y] = 1
    return y_onehot

y_tr_onehot = one_hot(y_tr, num_classes=4)

# -------------------------------------------------
# NUMPY MULTICLASS LOGISTIC REGRESSION
# -------------------------------------------------
class MyMulticlassLogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.W             = None
        self.b             = None

    def softmax(self, z):
        # Numerical stability trick: subtract max before exp
        z     = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y_onehot, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        n_classes = y_onehot.shape[1]

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

        self.train_losses = []
        self.val_losses   = []
        self.val_accs     = []

        for epoch in range(self.epochs):

            # --- Forward pass ---
            scores        = X @ self.W + self.b
            probabilities = self.softmax(scores)

            # --- Cross-entropy training loss ---
            loss = -np.mean(
                np.sum(y_onehot * np.log(probabilities + 1e-8), axis=1)
            )
            self.train_losses.append(loss)

            # --- Gradients (beta_new = beta_old - gamma * gradient) ---
            error = probabilities - y_onehot
            dW    = (X.T @ error) / n_samples
            db    = np.mean(error, axis=0, keepdims=True)

            # --- Update weights ---
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            # --- Validation loss and accuracy ---
            if X_val is not None:
                val_probs = self.softmax(X_val @ self.W + self.b)
                y_val_onehot = one_hot(y_val, num_classes=n_classes)

                val_loss = -np.mean(
                    np.sum(y_val_onehot * np.log(val_probs + 1e-8), axis=1)
                )
                val_acc  = np.mean(
                    np.argmax(val_probs, axis=1) == y_val
                ) * 100

                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)

            # --- Print every 100 epochs ---
            if epoch % 100 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch:4d} | "
                          f"Train Loss: {loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val Acc: {val_acc:.2f}%")
                else:
                    print(f"Epoch {epoch:4d} | Train Loss: {loss:.4f}")

    def predict(self, X):
        probs = self.softmax(X @ self.W + self.b)
        return np.argmax(probs, axis=1)

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------
model = MyMulticlassLogisticRegression(
    learning_rate=0.1,
    epochs=100
)

model.fit(X_tr, y_tr_onehot, X_val=X_val, y_val=y_val)

# -------------------------------------------------
# EVALUATE ON VALIDATION SET
# -------------------------------------------------
y_val_pred = model.predict(X_val)
val_accuracy = np.mean(y_val_pred == y_val) * 100
print(f"\nValidation Accuracy: {val_accuracy:.2f}%")

# -------------------------------------------------
# EVALUATE ON TEST SET
# -------------------------------------------------
y_pred       = model.predict(X_test)
test_accuracy = np.mean(y_pred == y_test) * 100
print(f"Test Accuracy:       {test_accuracy:.2f}%")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# -------------------------------------------------
# PLOT TRAINING AND VALIDATION LOSS
# -------------------------------------------------
epochs_range = range(model.epochs)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, model.train_losses, label="Training Loss")
plt.plot(epochs_range, model.val_losses,   label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline: Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "baseline_train_val_loss.png")
plt.show()

# -------------------------------------------------
# PLOT VALIDATION ACCURACY
# -------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, model.val_accs, color="orange", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Baseline: Validation Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "baseline_val_accuracy.png")
plt.show()

# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Baseline NumPy Logistic Regression — Confusion Matrix")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "baseline_confusion_matrix.png")
plt.show()

print("\nSaved figures to:", FIGURES_DIR)
