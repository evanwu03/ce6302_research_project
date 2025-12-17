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
import json

# -------------------------------
# CONFIG
# -------------------------------
DATA_ROOT = r"C:/Users/haris/mrlEyes/mrlEyes_2018_01"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


# -------------------------------
# STEP 1: INDEX ALL IMAGES
# -------------------------------
all_samples = []
subjects = set()

for subject_dir in sorted(os.listdir(DATA_ROOT)):
    subject_path = os.path.join(DATA_ROOT, subject_dir)

    if not os.path.isdir(subject_path):
        continue

    subject_id = subject_dir  # e.g., s0001
    subjects.add(subject_id)

    for fname in sorted(os.listdir(subject_path)):
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
subjects = sorted(subjects)
random.Random(SEED).shuffle(subjects)

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
def balance_split(samples, shuffle=True):
    open_eyes = [s for s in samples if s["label"] == 1]
    closed_eyes = [s for s in samples if s["label"] == 0]

    min_count = min(len(open_eyes), len(closed_eyes))

    if(shuffle):
      open_eyes = random.sample(open_eyes, min_count)
      closed_eyes = random.sample(closed_eyes, min_count)
      balanced = open_eyes + closed_eyes
      random.shuffle(balanced)
   
    else:
      open_eyes = open_eyes[:min_count]
      closed_eyes = closed_eyes[:min_count] 
      balanced = open_eyes + closed_eyes
      random.Random(SEED).shuffle(balanced)   

    return balanced

balanced_splits = {}
for split_name, samples in splits.items():
    if split_name in ["train", "val"]:
        balanced = balance_split(samples, shuffle=True)
    else:  # test split: no shuffle, always same
        balanced = balance_split(samples, shuffle=False)
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

with open('history.json', 'w') as f:
    json.dump(history.history, f)

# ----------------------------
# EVALUATE
# ----------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save("eye_state_custom_cnn_v6.h5")
print("Model saved as 'eye_state_custom_cnn_v6.h5'")





# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model("./eye_state_custom_cnn_v6.h5")

# -----------------------------
# Load training history
# -----------------------------
with open('history.json', 'r') as f:
    history = json.load(f)

train_loss = history.get('loss')
val_loss = history.get('val_loss')
train_accuracy = history.get('accuracy') or history.get('acc')
val_accuracy = history.get('val_accuracy') or history.get('val_acc')
epochs = np.arange(1, len(train_loss) + 1)

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(epochs, train_accuracy, 'o-', label='Train Accuracy')
plt.plot(epochs, val_accuracy, 's-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy per Epoch')
plt.xticks(epochs)
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# Plot Loss
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, 'o-', label='Train Loss')
plt.plot(epochs, val_loss, 's-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss per Epoch')
plt.xticks(epochs)
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# Combined Accuracy & Loss
# -----------------------------
fig, ax1 = plt.subplots(figsize=(8,5))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color='tab:blue')
ax1.plot(epochs, train_accuracy, 'o-', label='Train Accuracy', color='tab:blue')
ax1.plot(epochs, val_accuracy, 's-', label='Validation Accuracy', color='tab:cyan')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(epochs)
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel('Loss', color='tab:red')
ax2.plot(epochs, train_loss, 'o--', label='Train Loss', color='tab:red')
ax2.plot(epochs, val_loss, 's--', label='Validation Loss', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.9))
plt.title('Training and Validation Metrics')
plt.show()

# -----------------------------
# Evaluate on test dataset
# -----------------------------
y_true = []
probs_all = []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0).flatten()
    probs_all.extend(probs)
    y_true.extend(labels.numpy().flatten())

y_true = np.array(y_true)
probs_all = np.array(probs_all)
y_pred = (probs_all > 0.5).astype(int)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Closed","Open"], yticklabels=["Closed","Open"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Eye State CNN")
plt.show()

# -----------------------------
# Classification Report & Recall
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Closed", "Open"]))

closed_recall = recall_score(y_true, y_pred, pos_label=0)
print(f"Closed-eye Recall: {closed_recall:.4f}")

# -----------------------------
# ROC Curve & AUC
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_true, probs_all)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Eye State CNN')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

