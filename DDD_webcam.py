import cv2
import dlib
import numpy as np
from skimage.feature import hog
import joblib
from collections import deque

# ----------------------------
# Load trained SVM model
# ----------------------------
svm = joblib.load("drowsiness_svm.pkl")

# ----------------------------
# HOG parameters (same as training)
# ----------------------------
hog_params = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}

# ----------------------------
# Initialize dlib face detector
# ----------------------------
detector = dlib.get_frontal_face_detector()

# ----------------------------
# Webcam capture
# ----------------------------
cap = cv2.VideoCapture(0)

# ----------------------------
# Smoothing buffer
# ----------------------------
buffer_size = 10  # number of frames to smooth over
pred_buffer = deque(maxlen=buffer_size)

def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (64, 64))
    face_eq = cv2.equalizeHist(face_resized)
    features = hog(face_eq, **hog_params)
    return features.reshape(1, -1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for rect in faces:
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        pad = int(0.1 * (x2 - x1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)

        face_crop = gray[y1:y2, x1:x2]
        feat = preprocess_face(face_crop)
        pred = svm.predict(feat)[0]

        # --- Add prediction to buffer ---
        pred_buffer.append(pred)
        # Majority vote / most frequent prediction
        smoothed_pred = max(set(pred_buffer), key=pred_buffer.count)

        # Draw bounding box and label
        label = "Awake" if smoothed_pred == 0 else "Drowsy"
        color = (0, 255, 0) if smoothed_pred == 0 else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
