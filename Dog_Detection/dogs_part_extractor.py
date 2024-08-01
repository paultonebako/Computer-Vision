# Script for identifying frames in a video

import os
# Removes TF Logging. 0 prints all, 1 removes INFO, 2 removes INFO & WARNINGS, 3 removes all
import matplotlib.pyplot as plt


import numpy as np
import argparse
import time
import cv2

import tensorflow as tf
import tensorflow_hub as tfhub

from parameters import ParameterSet
from detectObject_noBoxes import run_detector




if __name__ == '__main__':
    # Parameter Handler
    params = ParameterSet()

    # Disable debug option in parameters to allow for cmdline arguments
    if not params.debug:
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", required=True, help="Directory to desired video")
        ap.add_argument("-o", "--video_output_dir", required=False, help="Output directory for labeled video.")
        args = vars(ap.parse_args())

        params.video_dir = args["video"]
        if args["video_output_dir"] is not None:
            params.output_vid_dir = args["video_output_dir"]
    else:
        # Designate video directory here if in DEBUG

        #params.video_dir = "videos/TimKing_Lulu Dev Video 01.m4v"
        params.video_dir = "videos/toloka2.mp4"

        print("[INFO] DEBUGMODE: Using Default Directories")

    # Initiate video reading checks
    cap = cv2.VideoCapture(params.video_dir)
    if cap is None or not cap.isOpened():
        print("[ERROR] Video file: [", params.video_dir, "] does not exist or is unreadable.")
        exit(0)
    else:
        # Get video dimensions (can use fps to determine frame skips)
        params.video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        params.video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)

    print("output location" , params.output_vid_dir)

    print("[INFO] Loading Detector...")
    detector = tfhub.load(params.detector_loc)
    print("[INFO] Detector Loaded.")

    print("[INFO] Loading & Modifying Video...")
    # Initiating video writer to rewrite images with obtained information
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(params.output_vid_dir, fourcc, 20.0,
                          (params.video_w, params.video_h), True)
    
    start_time = time.time()

    bad_frames = 0
    frame_num = 1
    
    #Video loading and modifying
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video steam, or error has occurred...")
            break
        # Video frames are read in using BGR, converted for use with keras
        co_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        or_frame = np.array(co_frame).copy()
        output, is_dog, boxes = run_detector(detector, co_frame, 0.75)
        # print(frame_num)
        # Detector in DetectObject returns a check on if the object is a dog, otherwise discards it.
        if is_dog:
            out_frame = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            out.write(out_frame)
        else:
            bad_frames += 1

        if params.show_live:
            cv2.imshow("Image", output)
            if cv2.waitKey(1) == ord('q'):
                break

        frame_num += 1

    # Release all video writers and readers
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    print("[INFO] Video Modification Complete")
    print("Elapsed time: ", (end_time - start_time), " seconds")

    # Small feature to compare predicted frames with their ground truth, can me modified above to make use of
    #print(bad_frames, "bad detections /", frame_num, "frames")
