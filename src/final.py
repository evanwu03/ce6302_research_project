import cv2
import numpy as np
import tensorflow as tf
import dlib
from scipy.spatial import distance
import time
from collections import deque

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = 32
MODEL_PATH = "eye_state_custom_cnn_v5.h5"
DROWSY_THRESHOLD = 2.0  # seconds
EAR_THRESHOLD = 0.25    # below this considered closed
CNN_WEIGHT = 0.6         # weighting for CNN probability
EAR_WEIGHT = 0.4         # weighting for EAR
ROLLING_WINDOW = 5       # frames

# Load CNN model
model = tf.keras.models.load_model(MODEL_PATH)

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
leye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
reye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")

# dlib face detector & landmarks (for EAR)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def preprocess_eye(eye_img):
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = gray.astype(np.float32)/255.0
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)
    return gray

def predict_eye_state(eye_img):
    preprocessed = preprocess_eye(eye_img)
    prob = model.predict(preprocessed, verbose=0)[0][0]
    state = "Open" if prob > 0.5 else "Closed"
    return state, prob

def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def shape_to_np(shape):
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

# ----------------------------
# WEBCAM LOOP
# ----------------------------
cap = cv2.VideoCapture(0)
closed_start_time = None

# Rolling average deque
score_history = deque(maxlen=ROLLING_WINDOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    dlib_faces = detector(gray_frame, 0)

    eyes_closed = False
    avg_ear = 1.0
    left_prob = 1.0
    right_prob = 1.0

    for (x, y, w, h) in faces:
        roi_face = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)


        # CNN eye detection
        left_eyes = leye_cascade.detectMultiScale(roi_face)
        for (ex, ey, ew, eh) in left_eyes:
            leye_img = roi_face[ey:ey+eh, ex:ex+ew]
            _, left_prob = predict_eye_state(leye_img)
            break

        right_eyes = reye_cascade.detectMultiScale(roi_face)
        for (ex, ey, ew, eh) in right_eyes:
            reye_img = roi_face[ey:ey+eh, ex:ex+ew]
            _, right_prob = predict_eye_state(reye_img)
            break

    # dlib EAR
    for d_face in dlib_faces:
        shape = predictor(gray_frame, d_face)
        shape = shape_to_np(shape)

        left_eye_pts = shape[LEFT_EYE_IDX]
        right_eye_pts = shape[RIGHT_EYE_IDX]
        avg_ear = (eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)) / 2.0
        break

    # Combine CNN and EAR with weighting
    combined_score = CNN_WEIGHT * ((left_prob + right_prob)/2) + EAR_WEIGHT * avg_ear
    score_history.append(combined_score)
    rolling_score = sum(score_history)/len(score_history)

    #cv2.putText(frame, f"Rolling Score: {rolling_score:.2f}", (50,40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    #cv2.putText(frame, f"Avg EAR: {avg_ear:.2f}", (50,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # Determine if drowsy
    if rolling_score < 0.6:
        eyes_closed = True
    else:
        eyes_closed = False

    # Drowsiness tracking
    current_time = time.time()
    if eyes_closed:
        if closed_start_time is None:
            closed_start_time = current_time
        elif current_time - closed_start_time >= DROWSY_THRESHOLD:
            cv2.putText(frame, "DROWSY!", (50,110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    else:
        closed_start_time = None

    cv2.imshow("Drowsiness Detection", frame)
    
    # If q key was pressed, break from loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
