# Test script to benchmark Dlib's facial landmark algortihm's accuracy against iBug-300W dataset 
# Author: Evan Wu
# Date: 11/5/2025
# Dataset script was tested with was sourced from: 
# https://www.kaggle.com/datasets/toxicloser/ibug-300w-large-face-landmark-dataset?resource=download-directory
# Note: Benchmarks was on a static set of images using bounding box ROIs manually described in the dataset 
# Using the dlib detector instead of predetermined bounding box gives ~2.46 NME overall


import xml.etree.ElementTree as ET
import time 
import cv2 as cv
import dlib
import os 
import numpy as np
import math
import matplotlib.pyplot as plt
from   utils import shape_to_np  


# keep track of time
start_time = time.time()
time_elapsed   = 0

# path of XML file
xml_path = "../ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml"

# dataset root folder 
dataset_root = os.path.dirname(xml_path)


# create XML tree
tree = ET.parse(xml_path)
root = tree.getroot()


# Store each datapoint as a tuple (image_path, box, landmarks (x,y)) as follows 
#{
#  "image": "path/to/image.jpg",
#  "box": {"left": 80, "top": 50, "width": 180, "height": 180},
#  "landmarks": [(x0, y0), ..., (x67, y67)]
#}
dataset = [
    {
        "image": image.attrib["file"],
        "box": {k: int(v) for k, v in box.attrib.items()},
        "landmarks": [(int(p.attrib["x"]), int(p.attrib["y"])) for p in box.findall("part")]
    }
    for image in root.iter("image")
    for box in image.findall("box")
]



imageSample = dataset[0] 
file = imageSample["image"]
box  = imageSample["box"]
landmarks = imageSample["landmarks"] 


total_images = len(dataset)
print(f"Total number of images: {total_images}")


# Procedure 
# ==========
# 1. Iterate through the dataset 
# 2. Load image from file path of current image
# 3. Run detector model to get face bounding box
# 4. Run landmark detection model for bounding box
# 5. Store predicted (x,y) coordinates
# 6. Calculate Normalized Mean Error 
# 7. Aggregate the mean NME across entire set 


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


# Aggregation of NME across entire dataset
all_nmes = [] 


for image in dataset:

    #print(image)
    frame = cv.imread(os.path.join(dataset_root, image["image"]))
    

    # Convert frame to grayscale 
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #rects = detector(gray, 0)
    b = image["box"]

    # Rectangles are manually assigned in dataset so we'll use those from XML file
    rects = dlib.rectangle(
                      b["left"],  b["top"],
                      b["left"] + b["width"],
                      b["top"] +  b["height"])
    
    
    shape = [] 

    # Bad detection skip to next sample
    if not rects: 
        print("Could not detect face")
        continue 


    #for rect in rects: 
    # determine the facial landmarks for face region the
    # convert the facial landmark (x,y) coordiantes to a NumPy
    # array
    shape = predictor(gray, rects)
    shape = shape_to_np(shape)

        
    # Not doing anything with vertices yet
    #x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    #cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    error = []
    parts = list(image["landmarks"])

    
    x36, y36 = parts[36] 
    x45, y45 = parts[45]
    interocular_dist = math.sqrt((int(x36) - int(x45))**2 + (int(y36) - int(y45))**2)

    for i in range(len(shape)):
        x_pred, y_pred = shape[i]
        x_gt, y_gt  = parts[i] 
        

        err = math.sqrt((int(x_pred) - int(x_gt))**2 + (int(y_pred) - int(y_gt))**2)
        error.append(err)
    

    # Normalize an image with computed interocular distance
    normalized_mean_err = (sum(error) / len(error)) / interocular_dist
    all_nmes.append(normalized_mean_err)

    #print(f"NME: {normalized_mean_err:.4f}")



# Compute statistics 
mean_nmes = np.mean(all_nmes)
std_nmes  = np.std(all_nmes)
print(f"Mean NME: {mean_nmes:.4f}, Std: {std_nmes:.4f}")

time_elapsed = time.time() - start_time 
print(f"Time elapsed: {time_elapsed:.2f}\n")

# Plot Cumulative Error Distribution
nmes_sorted = np.sort(all_nmes)
cum = np.arange(1, len(nmes_sorted)+1) / len(nmes_sorted)
plt.plot(nmes_sorted, cum, label="Predictor only")
plt.xlabel("Normalized Root Mean Error (NME)")
plt.ylabel("Probability (%)")
plt.title("Cumulative Error Distribution")
plt.grid(True)
plt.show()
