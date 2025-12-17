import tensorflow as tf
import numpy as np
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
import os
import random
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, recall_score

# -------------------------------
# CONFIG
# -------------------------------
DATA_ROOT = r"C:/Users/haris/mrlEyes/mrlEyes_2018_01"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# -------------------------------
# STEP 1: INDEX ALL IMAGES
# -------------------------------
all_samples = []
subjects = set()

for subject_dir in os.listdir(DATA_ROOT):
    subject_path = os.path.join(DATA_ROOT, subject_dir)

    if not os.path.isdir(subject_path):
        continue

    subject_id = subject_dir  # e.g., s0001
    subjects.add(subject_id)

    for fname in os.listdir(subject_path):
        if not fname.endswith(".png"):
            continue

        parts = fname.split("_")
        if len(parts) < 7:
            continue  # skip malformed files

        eye_state = int(parts[4])  # 0 = closed, 1 = open

        sample = {
            "subject": subject_id,
            "label": eye_state,
            "path": os.path.join(subject_path, fname)
        }

        all_samples.append(sample)

print(f"Total images indexed: {len(all_samples)}")
print(f"Total subjects: {len(subjects)}")

# -------------------------------
# STEP 2: SPLIT BY SUBJECT
# -------------------------------
subjects = list(subjects)
random.shuffle(subjects)

n_total = len(subjects)
n_train = int(TRAIN_RATIO * n_total)
n_val = int(VAL_RATIO * n_total)

train_subjects = set(subjects[:n_train])
val_subjects = set(subjects[n_train:n_train + n_val])
test_subjects = set(subjects[n_train + n_val:])

def assign_split(sample):
    if sample["subject"] in train_subjects:
        return "train"
    elif sample["subject"] in val_subjects:
        return "val"
    else:
        return "test"

splits = {"train": [], "val": [], "test": []}

for s in all_samples:
    split = assign_split(s)
    splits[split].append(s)

# -------------------------------
# STEP 3: BALANCE OPEN / CLOSED PER SPLIT
# -------------------------------
def balance_split(samples):
    open_eyes = [s for s in samples if s["label"] == 1]
    closed_eyes = [s for s in samples if s["label"] == 0]

    min_count = min(len(open_eyes), len(closed_eyes))

    open_eyes = random.sample(open_eyes, min_count)
    closed_eyes = random.sample(closed_eyes, min_count)

    balanced = open_eyes + closed_eyes
    #random.shuffle(balanced)

    return balanced

balanced_splits = {}
for split_name, samples in splits.items():
    balanced = balance_split(samples)
    balanced_splits[split_name] = balanced

# -------------------------------
# STEP 4: REPORT STATS
# -------------------------------
def report(split_name, samples):
    open_cnt = sum(1 for s in samples if s["label"] == 1)
    closed_cnt = sum(1 for s in samples if s["label"] == 0)
    subjects = set(s["subject"] for s in samples)

    print(f"\n{split_name.upper()}")
    print(f"  Images: {len(samples)}")
    print(f"  Open: {open_cnt}")
    print(f"  Closed: {closed_cnt}")
    print(f"  Subjects: {len(subjects)}")

for split_name, samples in balanced_splits.items():
    report(split_name, samples)

# -------------------------------
# FINAL OUTPUT
# -------------------------------
train_data = balanced_splits["train"]
val_data = balanced_splits["val"]
test_data = balanced_splits["test"]


# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

# ----------------------------
# PREPROCESS + AUGMENTATION
# ----------------------------
def preprocess_image(path, train=True):
    # Load grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.astype(np.float32) / 255.0

    # Convert to (H,W,1)
    img = np.expand_dims(img, axis=-1)

    if train:
        # Augmentation (eye-safe)
        if random.random() < 0.5:
            img = tf.image.random_brightness(img, 0.2)
        if random.random() < 0.5:
            img = tf.image.random_contrast(img, 0.8, 1.2)

    return img

# ----------------------------
# DATASET
# ----------------------------
def build_dataset(samples, train=True):
    paths = [s["path"] for s in samples]
    labels = [s["label"] for s in samples]

    def generator():
        for p, y in zip(paths, labels):
            yield preprocess_image(p, train), y

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        ),
    )

    if train:
        ds = ds.shuffle(1000)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ----------------------------
# LOAD DATASETS
# ----------------------------
# Make sure train_data, val_data, test_data are defined as lists of dicts:
# [{"path": "full_path_to_image.png", "label": 0/1}, ...]
train_ds = build_dataset(train_data, train=True)
val_ds   = build_dataset(val_data, train=False)
test_ds  = build_dataset(test_data, train=False)



# ----------------------------
# CUSTOM CNN MODEL
# ----------------------------
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# ----------------------------
# COMPILE
# ----------------------------
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# TRAIN
# ----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ----------------------------
# EVALUATE
# ----------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save("eye_state_custom_cnn_v5.h5")
print("Model saved as 'eye_state_custom_cnn_v5.h5'")

