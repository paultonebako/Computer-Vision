# Script for identifying frames in a video
# this code will classify dogs with bounding box showing the class name over the bounding box 
import os
# Removes TF Logging. 0 prints all, 1 removes INFO, 2 removes INFO & WARNINGS, 3 removes all
import matplotlib.pyplot as plt


import numpy as np
import argparse
import time
import cv2

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as tfhub
import keras
from parameters import ParameterSet
from DetectObject import run_detector


def LoadModels(num_models, params):
    models = []
    try:
        print("[INFO] Loading Network Model(s)...")
        for i in range(num_models):
            if num_models == 1:
                model_name = params.one_model
            else:
                model_name = params.model_dir + "model" + str(i+1) + ".h5"
            model = keras.models.load_model(model_name) #, custom_objects={'KerasLayer': tfhub.KerasLayer,
                                                                 #'TripletSemiHardLoss': tfa.losses.TripletSemiHardLoss})
            print(model.summary())
            models.append(model)

        print("[INFO] " + str(num_models) + " Model(s) Loaded.")
    except IOError as thisError:
        print("[ERROR] Model file: [", model_name, "] (or other model files) does not exist or is unreadable.")
        print(thisError)
        exit(0)

    return models


# Preprocess and classify pose of dog in image
def PredictThisImage(image, models, params):
    classes = ['Down', 'Between', 'Straight' , 'Side', 'Up']

    image = tf.keras.preprocessing.image.smart_resize(image, (params.m_height, params.m_width))

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    preds = [model.predict(img_array) for model in models]
    preds = np.asarray(preds)
    pred_scores = tf.nn.softmax(preds) * 100
    np.set_printoptions(suppress=True)

    res_scores = np.squeeze(tf.keras.backend.eval(pred_scores))

    temp = []

    if len(preds[0][0]) == 1:
        temp.append(preds[0][0][0] * 100)
        temp.append((1 - (preds[0][0][0])) * 100)
        label = classes[np.argmax(temp)]
        return label, temp

    
    if len(models) > 1:
        res_scores = np.average(res_scores, axis=0)
    

    label = classes[np.argmax(res_scores)]
    #print(res_scores)

    return label, res_scores



if __name__ == '__main__':
    # Parameter Handler
    params = ParameterSet()
    frames_folder = "output_frames"
    model = LoadModels(1, params)

    class_names = ['Down', 'Between', 'Side' , 'Straight', 'Up']

    try:
        os.mkdir("output_videos")
        os.mkdir(frames_folder)
    except :
        pass

    # Disable debug option in parameters to allow for cmdline arguments
    if not params.debug:
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", required=True, help="Directory to desired videos")
        ap.add_argument("-o", "--video_output_dir", required=False, help="Output directory for labeled video.")
        args = vars(ap.parse_args())

        params.video_dir = args["video"]
        if args["video_output_dir"] is not None:
            params.output_vid_dir = args["video_output_dir"]
    else:
        # Designate video directory here if in DEBUG

        #params.video_dir = "videos/TimKing_Lulu Dev Video 01.m4v"
        params.video_dir = "videos/"

        print("[INFO] DEBUGMODE: Using Default Directories")

    # we need to load the detector only once
    print("[INFO] Loading Detector...")
    detector = tfhub.load(params.detector_loc)
    print("[INFO] Detector Loaded.")

    listofvideos = os.listdir(params.video_dir)
    print(len(listofvideos))
    for vid_name in listofvideos:
        # Initiate video reading checks
        print("[INFO] Loading Video..." + vid_name)
        vid_dir = params.video_dir +"/" + vid_name
        cap = cv2.VideoCapture(vid_dir)
        if cap is None or not cap.isOpened():
            print("[ERROR] Video file: [", params.video_dir, "] does not exist or is unreadable.")
            exit(0)
        else:
            # Get video dimensions (can use fps to determine frame skips)
            params.video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            params.video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = cap.get(cv2.CAP_PROP_FPS)



        print("[INFO] Loading & Modifying Video...")
        # Initiating video writer to rewrite images with obtained information
        out_put_video_name =  vid_dir.split("/")[-1].split(".")[0]

        print("output location" , out_put_video_name)

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter("output_videos/"+out_put_video_name+".avi" , fourcc, 20.0,
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
            output, is_dog, boxes  = run_detector(detector, co_frame, 0.75)
            # print(frame_num)
           
            # Detector in DetectObject returns a check on if the object is a dog, otherwise discards it.
            if is_dog:
                print(boxes)
                xmin = boxes[1] * params.video_w
                ymin = boxes[0] * params.video_h
                print(xmin,ymin)
                out_frame = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                label, scores = PredictThisImage(or_frame, model, params)

                if np.max(scores) > 0.8:
                    #print("score ........")
                   # print(scores)
                    pred_class = class_names[np.argmax(scores)]
                else:
                    pred_class = 'Uncertain'
                cv2.putText(output, pred_class+ "  " + str(np.max(scores)) + "%" , (int(xmin),int(ymin)) , cv2.FONT_HERSHEY_COMPLEX , 0.8 , (255, 0, 0, 255) , 1)
                if params.save_frame :
                    cv2.imwrite(frames_folder+"/" + pred_class + "_" + out_put_video_name +"_fn"+str(frame_num)+".jpg", out_frame)
                out.write(out_frame)

               
            else:
          
                bad_frames += 1
            
            if params.show_live:
                cv2.imshow("Image", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
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
