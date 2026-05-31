from pathlib import Path
import numpy as np
from PIL import Image

# -------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
IMG_SIZE = (64, 64)

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
CLASS_TO_LABEL = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "notumor": 3,
}

# -------------------------------------------------
# HELPER FUNCTION
# -------------------------------------------------
def load_images_from_folder(base_dir, img_size=(64, 64)):
    X = []
    y = []

    for class_name in CLASS_NAMES:
        class_dir = base_dir / class_name
        label = CLASS_TO_LABEL[class_name]

        for img_path in class_dir.glob("*"):
            try:
                img = Image.open(img_path).convert("L")   # grayscale
                img = img.resize(img_size)
                img_array = np.array(img, dtype=np.float32) / 255.0

                X.append(img_array)
                y.append(label)
            except Exception as e:
                print(f"Could not read {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    return X, y

# -------------------------------------------------
# LOAD TRAIN AND TEST DATA
# -------------------------------------------------
print("Loading training data...")
X_train, y_train = load_images_from_folder(TRAIN_DIR, IMG_SIZE)

print("Loading testing data...")
X_test, y_test = load_images_from_folder(TEST_DIR, IMG_SIZE)

# -------------------------------------------------
# PRINT SHAPES
# -------------------------------------------------
print("\n--- DATA SHAPES ---")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

print("\nExample label mapping:")
for k, v in CLASS_TO_LABEL.items():
    print(f"{k:12s} -> {v}")

# -------------------------------------------------
# SAVE ARRAYS
# -------------------------------------------------
np.save(MODELS_DIR / "X_train.npy", X_train)
np.save(MODELS_DIR / "y_train.npy", y_train)
np.save(MODELS_DIR / "X_test.npy", X_test)
np.save(MODELS_DIR / "y_test.npy", y_test)

print("\nSaved prepared arrays in:")
print(MODELS_DIR)