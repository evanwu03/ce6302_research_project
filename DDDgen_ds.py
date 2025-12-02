import os
import cv2
import numpy as np
from skimage.feature import hog
import joblib

# ----------------------------
# Paths
# ----------------------------
clean_base = os.path.expanduser("~/UTA-RLDD_clean")
save_path = os.path.expanduser("~/UTA-RLDD_HOG_dataset.npz")  # will save X, y

# Folds/parts to process
folds = ["f2p1", "f2p2", "f3p1", "f3p2", "f4p1", "f4p2", "f5p1", "f5p2"]

# ----------------------------
# HOG parameters
# ----------------------------
hog_params = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}

# ----------------------------
# Initialize lists
# ----------------------------
X = []
y = []

# ----------------------------
# Loop through folds, participants, and states
# ----------------------------
for fold in folds:
    fold_path = os.path.join(clean_base, fold)
    if not os.path.isdir(fold_path):
        print(f"Warning: {fold_path} does not exist, skipping.")
        continue

    for participant in sorted(os.listdir(fold_path)):
        participant_path = os.path.join(fold_path, participant)
        if not os.path.isdir(participant_path):
            continue

        for state in ["drowsy", "non-drowsy"]:
            state_path = os.path.join(participant_path, state)
            if not os.path.isdir(state_path):
                continue

            for img_file in sorted(os.listdir(state_path)):
                img_path = os.path.join(state_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Unreadable image: {img_path}")
                    continue

                # Compute HOG features
                features = hog(img, **hog_params)
                X.append(features)
                y.append(1 if state == "drowsy" else 0)

# ----------------------------
# Convert to arrays and save
# ----------------------------
X = np.array(X)
y = np.array(y)

np.savez_compressed(save_path, X=X, y=y)
print(f"HOG dataset saved to {save_path}")
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
