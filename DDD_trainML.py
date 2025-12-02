# train_svm.py
import numpy as np
from sklearn.svm import SVC
import joblib
import os


file_path = os.path.expanduser("~/UTA-RLDD_HOG_dataset.npz")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}")

data = np.load(file_path)
X = data["X"]
y = data["y"]


print("Dataset loaded successfully!")
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)


print("\n Training SVM (kernel='linear', C=1) on full dataset...")
svm_model = SVC(kernel="linear", C=1)
svm_model.fit(X, y)


model_path = os.path.expanduser("~/drowsiness_svm.pkl")
joblib.dump(svm_model, model_path)
print(f"\n Model trained and saved successfully to: {model_path}")
