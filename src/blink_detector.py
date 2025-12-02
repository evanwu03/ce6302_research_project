


import cv2 as cv
import numpy as np
import time 
import argparse
import imutils
import dlib 
from scipy.spatial import distance as dist

# User-defined files
from utils import shape_to_np  
from imutils import face_utils



# Computes the eye aspect ratio of one eye as described in 
# Real-Time Eye Blink Detection Using Facial Landmarks (Soukupová, Čech 2016)
def eye_aspect_ratio(eye): 
    
    # Compute euclidean distance of two sets of vertical eye landmarks (x,y)-coords
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute euclidean distance between horizontal eye landmarks (x,y)-coods
    C = dist.euclidean(eye[0], eye[3])

    # Compute eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # Return eye aspect ratio
    return ear
    

# Opens a video stream on webcam, detects eye facial landmarks and computes the EAR, using a threshold value 
# to determine if the person in the video has blinked. Every time a blink is detected, a counter is incremented
def blink_detector(): 

    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0


    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Set resolution 
    ret = cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    ret = cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)   
    
    # FPS counter
    # used to record the time when we processed last frame
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0


    # construct the argument parser and parse the arguments 
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    args = vars(ap.parse_args())



    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])


    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



    while True: 
        # Capture Frame by Frame
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=500)


        # if frame is read correctly ret is True
        if not ret: 
            print("Can't receive frame (stream end?). Exiting...")
            break


        # Convert frame to grayscale 
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect faces in grayscale image
        rects = detector(gray, 0) # (image, number of image pyramid layers for upsampling)

        # Display current status of Detector
        """ if not rects: 
             cv.putText(frame, "Lost Tracking", (190, 30),
             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else: 
            cv.putText(frame, "Tracking", (190, 30),
            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) """
        

        for rect in rects: 
            # determine the facial landmarks for face region the
            # convert the facial landmark (x,y) coordiantes to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)


            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
        

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv.convexHull(leftEye)
            rightEyeHull = cv.convexHull(rightEye)
            cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)




            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                # reset the eye frame counter
                COUNTER = 0
            


            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



            # Draws bounding box over face
            """ x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) """


            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on image
            for (x, y) in shape:
                cv.circle(frame, (x, y), 1, (0, 0, 255), -1)
        

        # Calculate FPS
        new_frame_time = time.time()

        fps = 1 / (new_frame_time - prev_frame_time) 
        prev_frame_time = new_frame_time

    
        # show the frame 
        # Display the resulting frame
        # Draw FPS on frame
        #cv.putText(frame, f"FPS: {fps:.2f}", (10, 30),
        #cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    
        cv.imshow('frame', frame)

        # If q key was pressed, break from loop
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break



    # Release frame and destroy window
    cap.release()
    cv.destroyAllWindows()


