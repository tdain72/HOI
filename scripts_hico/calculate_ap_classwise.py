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
import json
with open('../infos/directory.json') as fp:
    all_data_dir = json.load(fp)
with open(all_data_dir + 'hico_infos/hico_list_vb.json') as fp:
    list_vb = json.load(fp)
VERB2ID = list_vb
NO_VERBS = 117
VERBS_NO_COCO = 80

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
    count = 0
    for k in range(NO_VERBS):
        if VERB2ID[str(k + 1)]:
            predicted = predicted_score[:, k]
            true = true_score[:, k]
            try:
                AP_s = average_precision_score(true, predicted) * 100
            except:

                import pdb
                pdb.set_trace()
            if np.isnan(AP_s):
                pass
            else:
                mean += AP_s
                count = count + 1

            result.append((VERB2ID[str(k + 1)], AP_s))

    result.append(('Mean', mean / count))
    mean = 0.0
    counter = 0
    return (result, [('AP', average_precision_score(true_single_class,
            predicted_single_class) * 100)])


if __name__ == '__main__':
    predicted_score = np.random.random_sample([10, 117])
    true_score = np.random.randint(2, size=(10, 117))
    predicted_score_single = np.random.random_sample([10, 1])
    true_score_single = np.random.randint(2, size=(10, 1))
    final = class_AP(predicted_score, true_score,
                     predicted_score_single, true_score_single)
    print(final)
