#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import cv2
import pandas as pd
import json
import helpers_preprocess as helpers_pre

with open('../infos/directory.json') as fp:
    all_data_dir = json.load(fp)

OBJ_PATH_train_s = all_data_dir + 'Object_Detections_hico/train/'
OBJ_PATH_test_s = all_data_dir + 'Object_Detections_hico/val/'
image_dir_train = all_data_dir + 'Data_hico/train2015'
image_dir_test = all_data_dir + 'Data_hico/test2015'

VERB2ID = {
    'carry': 0,
    'catch': 1,
    'cut_instr': 2,
    'cut_obj': 3,
    'drink': 4,
    'eat_instr': 5,
    'eat_obj': 6,
    'hit_instr': 7,
    'hit_obj': 8,
    'hold': 9,
    'jump': 10,
    'kick': 11,
    'lay': 12,
    'look': 13,
    'point': 14,
    'read': 15,
    'ride': 16,
    'run': 17,
    'sit': 18,
    'skateboard': 19,
    'ski': 20,
    'smile': 21,
    'snowboard': 22,
    'stand': 23,
    'surf': 24,
    'talk_on_phone': 25,
    'throw': 26,
    'walk': 27,
    'work_on_computer': 28,
    }

ID2VERB = dict((VERB2ID[i], i) for i in VERB2ID)

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

pd.options.display.max_columns = 250  # None -> No Restrictions
pd.options.display.max_rows = 200  # None -> Be careful with


def visual(
    image_id,
    flag,
    pairs_info,
    score_HOI,
    score_interact,
    score_obj_box,
    score_per_box,
    score_REL,
    score_HOI_pair,
    ground_truth,
    ):
    start = 0
    for batch in range(len(image_id)):
        this_image = int(image_id[batch])
        a = helpers_pre.get_compact_detections(this_image, flag)
        person_bbxn = a['person_bbx']
        obj_bbxn = a['objects_bbx']
        this_batch_pers = int(pairs_info[batch][0])
        this_batch_objs = int(pairs_info[batch][1])
        increment = this_batch_pers * this_batch_objs
        ground_truth_this_batch = ground_truth[start:start + increment]
        score_HOI_this_batch = score_HOI[start:start + increment]
        start += increment
        if flag == 'train':

            cur_obj_path_s = OBJ_PATH_train_s \
                + 'HICO_train2015_%.8i.json' % this_image

            image_dir_s = image_dir_train + '/HICO_train2015_%.8i.json' \
                % this_image
        elif flag == 'test':

            cur_obj_path_s = OBJ_PATH_test_s \
                + 'HICO_train2015_%.8i.json' % this_image
            image_dir_s = image_dir_test + '/HICO_train2015_%.8i.json' \
                % this_image
        with open(cur_obj_path_s) as fp:
            detections = json.load(fp)
        img_H = detections['H']
        img_W = detections['W']
        person_bbx = np.array([img_W, img_H, img_W, img_H],
                              dtype=float) * person_bbxn
        obj_bbx = np.array([img_W, img_H, img_W, img_H], dtype=float) \
            * obj_bbxn
        img = cv2.imread(image_dir_s, 3)
        start_index = 0
        for person_box in person_bbx:
            for object_box in obj_bbx:
                ground_truth_this_sample = \
                    ground_truth_this_batch[start_index]
                score_HOI_this_sample = \
                    score_HOI_this_batch[start_index]
                print(score_HOI_this_sample)
                pred = [('GROUND_TRUTH', [(ID2VERB[ind], float('%.2f'
                        % ground_truth_this_sample[ind])) for ind in
                        np.argsort(ground_truth_this_sample)[-5:][::
                        -1]])]
                pred.append(('TOTAL_PREDICTION', [(ID2VERB[ind],
                            float('%.2f' % score_HOI_this_sample[ind]))
                            for ind in
                            np.argsort(score_HOI_this_sample)[-5:][::
                            -1]]))
                prediction = pd.DataFrame(pred, columns=['Name',
                        'Prediction'])

                img = cv2.imread(image_dir_s, 3)
                (x, y, w, h) = (int(person_box[0]), int(person_box[1]),
                                int(person_box[2] - person_box[0]),
                                int(person_box[3] - person_box[1]))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0),
                              3)
                (x, y, w, h) = (int(object_box[0]), int(object_box[1]),
                                int(object_box[2] - object_box[0]),
                                int(object_box[3] - object_box[1]))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255),
                              3)

                print('''
Predictions (Five Highest Confidence Class):
{}
'''.format(prediction))

                cv2.imshow('image', img)
                start_index += 1
                k = cv2.waitKey(0)
                if k == 27:  # wait for ESC key to exit

                    cv2.destroyAllWindows()

            if k == 27:  # wait for ESC key to exit

                cv2.destroyAllWindows()
        if k == 27:  # wait for ESC key to exit

            cv2.destroyAllWindows()

    cv2.destroyAllWindows()
