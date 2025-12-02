

import cv2 as cv
import numpy as np
import time 
import argparse
import imutils
import dlib 



from utils import shape_to_np  

from blink_detector import blink_detector

def main(): 
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


    while True: 
        # Capture Frame by Frame
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=400)


        # if frame is read correctly ret is True
        if not ret: 
            print("Can't receive frame (stream end?). Exiting...")
            break


        # Convert frame to grayscale 
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect faces in grayscale image
        rects = detector(gray, 0) # (image, number of image pyramid layers for upsampling)

        # Display current status of Detector
        if not rects: 
             cv.putText(frame, "Lost Tracking", (190, 30),
             cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else: 
            cv.putText(frame, "Tracking", (190, 30),
            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

        for rect in rects: 
            # determine the facial landmarks for face region the
            # convert the facial landmark (x,y) coordiantes to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


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
        cv.putText(frame, f"FPS: {fps:.2f}", (10, 30),
        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
        cv.imshow('frame', frame)


    # If q key was pressed, break from loop
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break



    # Release frame and destroy window
    cap.release()
    cv.destroyAllWindows()



if __name__ == "__main__": 
    #main()
    blink_detector()


