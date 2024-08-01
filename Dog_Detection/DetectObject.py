# Object Detection for TF2

import os
# Removes TF Logging. 0 prints all, 1 removes INFO, 2 removes INFO & WARNINGS, 3 removes all
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import torch

import numpy as np
from PIL import Image, ImageDraw

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, thickness=4):
    image_pil = None
    
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(np.uint8(image), mode = "RGB")
    elif isinstance(image, Image):
        image_pil = image.copy()
    
    draw = ImageDraw.Draw(image_pil)
    
    draw.rectangle([xmin, ymin, xmax, ymax], 
                width=thickness, 
                outline=(81,50,194,255))
    #print(image_pil)
    return np.asarray(image_pil)

def draw_boxes(boxes, class_ids, scores, min_score, cl = 18):
    class_ids = class_ids.astype(np.uint8)
    Positive_Scores = []
    All_Scores = []

    positive_id = class_ids==cl
    above_threshold = scores >= min_score

    out = boxes[(above_threshold) & (positive_id), :]
    Positive_Scores = scores[(positive_id) & (above_threshold)]
    All_Scores = scores[(positive_id)]
    
    #for i in range(boxes.shape[0]):
    #    if scores[i] >= min_score and int(class_ids[i]) == cl:
    #        # Check if the class_id is 18, since COCO 2017 was used for pretraining, 18 = 'dog'
    #        out.append(boxes[i])
    #        clsScores.append(scores[i]))
    #
    #    if int(class_ids[i]) == cl:
    #        All_Scores.append(scores[i])
        

    # return the boxes, if anything was detected
    return out, out.size > 0, Positive_Scores, All_Scores


def run_detector(detector, image, min_score):

    converted_img = tf.image.convert_image_dtype(image, tf.uint8)[tf.newaxis, ...]

    result = detector(converted_img)

    max_checks = 10
    #print(result)
    # Get the classIds from the most relevant result and convert them from a tensor to a list
    class_ids_tensor = result["detection_classes"][0]
    class_ids = tf.keras.backend.eval(class_ids_tensor)[:max_checks]
    class_ids = class_ids.astype(np.uint8)

    # Repeat for class confidence scores
    class_scores_tensor = result["detection_scores"][0]
    class_scores = tf.keras.backend.eval(class_scores_tensor)[:max_checks]

    # Repeat for prediction bounding boxes
    class_boxes_tensor = result["detection_boxes"][0]
    all_boxes = tf.keras.backend.eval(class_boxes_tensor)[:max_checks]


    return draw_boxes(all_boxes, class_ids, class_scores, min_score)


def run_detector_YOLOv8(detector, image, min_score):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = detector(image, device=device)
    result = result[0].cpu()
    
    # Get the classIds from the most relevant result and convert them from a tensor to a list
    class_ids = result.boxes.cls.numpy()

    # Repeat for class confidence scores
    class_scores = result.boxes.conf.numpy()

    # Repeat for prediction bounding boxes
    #translate from xyxy to yxyx
    all_boxes = result.boxes.xyxy.numpy()[:, [1,0,3,2]]

    return draw_boxes(boxes=all_boxes,
                      class_ids=class_ids,
                      scores=class_scores,
                      min_score=min_score,
                      cl = 16)

def ClassifySignals(PredictionSet):
    """
    Temp methhod detailing signals currently possible to classify based on 
    Image Classifier should be replaced by basic AI methods which eventually
    include time series analysis or a video (3dConv) based classifier.

    """
    SevereCowering = set(['close mouth', 'Inside', 'Ears_Back', 'Standing', 'Between'])
    Cowering = set(['close mouth', 'Inside', 'Ears_Back', 'Standing', 'Down'])

    Begging1 = set(['close mouth', 'Inside', 'Ears_Down', 'Sitting' ,'Down'])
    Begging2 = set(['close mouth', 'Inside', 'Ears_Down', 'Sitting' ,'Up'])

    Happy1 = set(['close mouth', 'Inside', 'Ears_Side', 'Sitting' ,'Up'])
    Happy2 = set(['open mouth', 'Inside', 'Ears_Side', 'Sitting' ,'Down'])
    
    Happy3 = set(['Lies_on_floor' ,'Straight'])
    Happy4 = set(['Lies_on_floor' ,'Down'])

    Please = set(['close mouth', 'Inside', 'Ears_Side', 'Sitting' ,'Down'])

    Alert = set(['close mouth', 'Inside', 'Up', 'Standing'])


    if PredictionSet.issubset(Cowering):
        return 'Fear/Afraid'
    elif PredictionSet.issubset(SevereCowering):
        return 'Fear/Afraid'
    elif PredictionSet.issubset(Begging1) or PredictionSet.issubset(Begging2):
        return 'Begging'
    elif PredictionSet.issubset(Happy1) or PredictionSet.issubset(Happy2):
        return 'Happy'
    elif PredictionSet.issubset(Please):
        return 'Please'
    elif PredictionSet.issuperset(Happy3) or PredictionSet.issuperset(Happy4):
        return 'Happy'
    else:
        return None

         