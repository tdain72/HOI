#!/usr/bin/python
# -*- coding: utf-8 -*-
#### This script will calculate AP per classwise without considering bounding boxes precision. This can be used to have a sanity check on the learnt model###

from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
import pandas as pd
import torch
import numpy as np
import os
import random
NO_VERBS = 29
VERBS_NO_COCO = 80

VERB2ID = [
    'carry',
    'catch',
    'cut_instr',
    'cut_obj',
    'drink',
    'eat_instr',
    'eat_obj',
    'hit_instr',
    'hit_obj',
    'hold',
    'jump',
    'kick',
    'lay',
    'look',
    'point',
    'read',
    'ride',
    'run',
    'sit',
    'skateboard',
    'ski',
    'smile',
    'snowboard',
    'stand',
    'surf',
    'talk_on_phone',
    'throw',
    'walk',
    'work_on_computer',
    ]

coco_verbs = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
    ]

threshold = 0.1


def class_AP(*args):
    result = []
    predicted_score = args[0]
    true_score = args[1]
    predicted_single_class = args[2]
    true_single_class = args[3]
    mean = 0
    for k in range(NO_VERBS):
        if VERB2ID[k]:
            predicted = predicted_score[:, k]
            true = true_score[:, k]
            try:
                AP_s = average_precision_score(true, predicted) * 100
            except:

                import pdb
                # pdb.set_trace()

            mean += AP_s
            result.append((VERB2ID[k], AP_s))

    result.append(('Mean', mean / NO_VERBS))
    mean = 0.0
    counter = 0
    return (result, [('AP', average_precision_score(true_single_class,
            predicted_single_class) * 100)])


if __name__ == '__main__':
    predicted_score = np.random.random_sample([10, 29])
    true_score = np.random.randint(2, size=(10, 29))
    predicted_score_single = np.random.random_sample([10, 1])
    true_score_single = np.random.randint(2, size=(10, 1))
    final = class_AP(predicted_score, true_score,
                     predicted_score_single, true_score_single)
    print(final)
