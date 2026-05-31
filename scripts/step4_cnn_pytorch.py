# ================================================
# 62533 Applied Machine Learning and Big Data
# Brain Tumour MRI Classification
# Author: Simin Asadi
# Study Number: s234955
# Date: May 2026
# ================================================

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import seaborn as sns

# -------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# LOAD PREPARED DATA
# -------------------------------------------------
X_train = np.load(MODELS_DIR / "X_train.npy")
y_train = np.load(MODELS_DIR / "y_train.npy")

X_test = np.load(MODELS_DIR / "X_test.npy")
y_test = np.load(MODELS_DIR / "y_test.npy")

print("Loaded data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)

# Add channel dimension for CNN: (N, 1, 64, 64)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

print("\nAfter adding channel dimension:")
print("X_train:", X_train.shape)
print("X_test: ", X_test.shape)

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]

# -------------------------------------------------
# CUSTOM DATASET
# -------------------------------------------------
class MRIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



full_train_dataset = MRIDataset(X_train, y_train)
test_dataset = MRIDataset(X_test, y_test)

val_size = int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")
print(f"Testing samples: {len(test_dataset)}")

# -------------------------------------------------
# CNN MODEL
# -------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 64x64 -> 64x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 64x64 -> 32x32

            nn.Conv2d(8, 16, kernel_size=3, padding=1), # 32x32 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 32x32 -> 16x16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),# 16x16 -> 16x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                           # 16x16 -> 8x8
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# -------------------------------------------------
# TRAINING
# -------------------------------------------------
EPOCHS = 100  #10 versus 100 versus 200

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

best_val_acc = 0.0
best_model_path = MODELS_DIR / "brain_tumor_cnn_best.pth"

for epoch in range(EPOCHS):
    # -----------------------------
    # TRAINING
    # -----------------------------
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    # -----------------------------
    # VALIDATION
    # -----------------------------
    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"- Train Loss: {epoch_loss:.4f} "
        f"- Train Acc: {epoch_acc:.2f}% "
        f"- Val Loss: {val_loss:.4f} "
        f"- Val Acc: {val_acc:.2f}%"
    )

print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
print("Best model saved to:", best_model_path)

# Load best model before final testing
model.load_state_dict(torch.load(best_model_path, map_location=device))

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
model_path = MODELS_DIR / "brain_tumor_cnn.pth"
torch.save(model.state_dict(), model_path)
print("\nSaved model to:", model_path)

# -------------------------------------------------
# TESTING
# -------------------------------------------------
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_accuracy = 100 * np.mean(all_preds == all_labels)
print(f"\nCNN Test Accuracy: {test_accuracy:.2f}%")

# -------------------------------------------------
# CLASSIFICATION REPORT
# -------------------------------------------------
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("CNN Confusion Matrix")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cnn_confusion_matrix.png")
plt.show()

print("\nSaved confusion matrix to:")
print(FIGURES_DIR / "cnn_confusion_matrix.png")

# -------------------------------------------------
# TRAINING CURVES
# -------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cnn_training_loss.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_accuracies, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Training Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "cnn_training_accuracy.png")
plt.show()

print("\nSaved training curves to:")
print(FIGURES_DIR / "cnn_training_loss.png")
print(FIGURES_DIR / "cnn_training_accuracy.png")
