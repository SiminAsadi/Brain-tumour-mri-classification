from pathlib import Path
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# PATHS
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
K_FOLDS = 5
EPOCHS_PER_FOLD = 30
FINAL_EPOCHS = 30
BATCH_SIZE = 16
PATIENCE = 5
RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# TRANSFORMS
# -------------------------------------------------
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),

    # Data augmentation
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# DATASETS
# -------------------------------------------------
train_dataset_aug = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
train_dataset_eval = datasets.ImageFolder(TRAIN_DIR, transform=eval_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

class_names = train_dataset_eval.classes
targets = np.array(train_dataset_eval.targets)

print("Classes:", class_names)
print("Class mapping:", train_dataset_eval.class_to_idx)

# -------------------------------------------------
# MODEL
# -------------------------------------------------
def create_resnet50_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze all pretrained layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final classification layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, len(class_names))
    )

    return model.to(device)

# -------------------------------------------------
# TRAIN ONE EPOCH
# -------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds) * 100

    return avg_loss, acc

# -------------------------------------------------
# EVALUATE
# -------------------------------------------------
def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds) * 100

    return avg_loss, acc, np.array(all_preds), np.array(all_labels)

# -------------------------------------------------
# K-FOLD CROSS-VALIDATION
# -------------------------------------------------
skf = StratifiedKFold(
    n_splits=K_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_accuracies = []

print("\nStarting fine-tuned ResNet50 K-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(
    skf.split(np.zeros(len(targets)), targets)
):
    print(f"\n========== Fold {fold + 1}/{K_FOLDS} ==========")

    train_subset = Subset(train_dataset_aug, train_idx)
    val_subset = Subset(train_dataset_eval, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = create_resnet50_model()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS_PER_FOLD):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch + 1}/{EPOCHS_PER_FOLD} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Train Acc: {train_acc:.2f}% "
            f"- Val Loss: {val_loss:.4f} "
            f"- Val Acc: {val_acc:.2f}%"
        )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    fold_accuracies.append(best_val_acc)

    fold_model_path = MODELS_DIR / f"resnet50_finetuned_fold_{fold + 1}.pth"
    torch.save(best_model_state, fold_model_path)

    print(f"Best validation accuracy fold {fold + 1}: {best_val_acc:.2f}%")
    print("Saved fold model:", fold_model_path)

# -------------------------------------------------
# K-FOLD SUMMARY
# -------------------------------------------------
print("\n========== K-FOLD RESULTS ==========")
print("Fold accuracies:", [round(acc, 2) for acc in fold_accuracies])
print(f"Mean validation accuracy: {np.mean(fold_accuracies):.2f}%")
print(f"Standard deviation: {np.std(fold_accuracies):.2f}%")

# -------------------------------------------------
# FINAL TRAINING ON FULL TRAINING DATA
# -------------------------------------------------
print("\nTraining final fine-tuned ResNet50 model on full training data...")

full_train_loader = DataLoader(
    train_dataset_aug,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

final_model = create_resnet50_model()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, final_model.parameters()),
    lr=1e-5
)

for epoch in range(FINAL_EPOCHS):
    train_loss, train_acc = train_one_epoch(
        final_model,
        full_train_loader,
        criterion,
        optimizer
    )

    print(
        f"Final Model Epoch {epoch + 1}/{FINAL_EPOCHS} "
        f"- Train Loss: {train_loss:.4f} "
        f"- Train Acc: {train_acc:.2f}%"
    )

final_model_path = MODELS_DIR / "resnet50_finetuned_final.pth"
torch.save(final_model.state_dict(), final_model_path)

print("\nSaved final model:", final_model_path)

# -------------------------------------------------
# FINAL TEST EVALUATION
# -------------------------------------------------
test_loss, test_acc, test_preds, test_labels = evaluate(
    final_model,
    test_loader,
    criterion
)

print("\n========== FINAL TEST RESULT ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        test_labels,
        test_preds,
        target_names=class_names
    )
)

# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
cm = confusion_matrix(test_labels, test_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Fine-tuned ResNet50 Confusion Matrix")
plt.tight_layout()

cm_path = FIGURES_DIR / "resnet50_finetuned_confusion_matrix.png"
plt.savefig(cm_path)
plt.show()

print("Saved confusion matrix to:", cm_path)
