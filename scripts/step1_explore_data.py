================================================
62533 Applied Machine Learning and Big Data
Brain Tumour MRI Classification
Author: Simin Asadi
Study Number: s234955
Date: May 2026
================================================

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random

# -------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
FIGURES_DIR = PROJECT_ROOT / "figures"

FIGURES_DIR.mkdir(exist_ok=True)

classes = ["glioma", "meningioma", "pituitary", "notumor"]

# -------------------------------------------------
# COUNT IMAGES
# -------------------------------------------------
print("\n--- DATASET OVERVIEW ---")

for split_name, split_path in [("Training", TRAIN_DIR), ("Testing", TEST_DIR)]:
    print(f"\n{split_name}:")
    total = 0

    for cls in classes:
        folder = split_path / cls
        count = len(list(folder.glob("*")))
        total += count
        print(f"{cls:12s}: {count}")

    print(f"Total images: {total}")

# -------------------------------------------------
# SHOW RANDOM SAMPLE IMAGES
# -------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle("Sample MRI Images", fontsize=16)

for row, split_path in enumerate([TRAIN_DIR, TEST_DIR]):
    for col, cls in enumerate(classes):
        folder = split_path / cls
        image_files = list(folder.glob("*"))

        if len(image_files) == 0:
            continue

        img_path = random.choice(image_files)

        img = Image.open(img_path)

        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].set_title(f"{cls}")
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "sample_images.png")
plt.show()

print("\nSaved figure:")
print(FIGURES_DIR / "sample_images.png")
