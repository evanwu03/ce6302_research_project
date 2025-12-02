import os
import cv2
import dlib

# Base paths
input_base = os.path.expanduser("~/UTA-RLDD")      # raw extracted frames
output_base = os.path.expanduser("~/UTA-RLDD_clean")    # cleaned + cropped faces
log_file_path = os.path.join(output_base, "skipped_log.txt")

# Folds/parts to process
folds = ["f2p2", "f3p1", "f3p2", "f4p1", "f4p2", "f5p1", "f5p2"]

# Create output directory if not exist
os.makedirs(output_base, exist_ok=True)

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

# Open log file
log_file = open(log_file_path, "w")

for fold in folds:
    fold_path = os.path.join(input_base, fold)
    if not os.path.isdir(fold_path):
        print(f"Warning: {fold_path} does not exist, skipping.")
        continue

    print(f"\nProcessing fold: {fold}")

    # Loop through participants in this fold
    for participant in sorted(os.listdir(fold_path)):
        participant_path = os.path.join(fold_path, participant)
        if not os.path.isdir(participant_path):
            continue

        print(f"  Participant: {participant}")

        for state in ["drowsy", "non-drowsy"]:
            input_folder = os.path.join(participant_path, state)
            output_folder = os.path.join(output_base, fold, participant, state)
            os.makedirs(output_folder, exist_ok=True)

            for img_file in sorted(os.listdir(input_folder)):
                img_path = os.path.join(input_folder, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    log_file.write(f"Unreadable: {img_path}\n")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)  # brightness normalization

                rects = detector(gray)

                if len(rects) != 1:
                    log_file.write(f"No/Multiple faces: {img_path}\n")
                    continue

                # Crop and resize to a consistent size
                rect = rects[0]
                x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()

                # Add a small margin to avoid tight cropping
                h, w = gray.shape
                pad = int(0.1 * (x2 - x1))
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

                face = gray[y1:y2, x1:x2]
                face_resized = cv2.resize(face, (64, 64))

                # Save clean face image
                output_path = os.path.join(output_folder, img_file)
                cv2.imwrite(output_path, face_resized)

log_file.close()
print("\nCleaning complete! All skipped files logged in:", log_file_path)
