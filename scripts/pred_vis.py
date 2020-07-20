#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import cv2
import pandas as pd
import json
import helpers_preprocess as helpers_pre
import grad_cam as gc
import torch
import dataloader_vcoco as data_loader
import matplotlib.pyplot as plt

with open('../infos/directory.json') as fp:
    all_data_dir = json.load(fp)

OBJ_PATH_train_s = all_data_dir + 'Object_Detections_vcoco/train/'
OBJ_PATH_test_s = all_data_dir + 'Object_Detections_vcoco/val/'
image_dir_train = all_data_dir + 'Data_vcoco/train2014'
image_dir_val = all_data_dir + 'Data_vcoco/train2014'
image_dir_test = all_data_dir + 'Data_vcoco/val2014'

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
    single_image,
    model,
    predict,
    union_box,
    rois_people,
    rois_object,
    context,
    ):
    plt.subplot()
    start = 0
    for batch in range(len(image_id)):
        this_image = int(image_id[batch])
        if this_image == int(single_image) and single_image != 'f':
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
                    + 'COCO_train2014_%.12i.json' % this_image

                image_dir_s = image_dir_train + '/COCO_train2014_%.12i.jpg' \
                    % this_image
            elif flag == 'test':

                cur_obj_path_s = OBJ_PATH_test_s \
                    + 'COCO_val2014_%.12i.json' % this_image
                image_dir_s = image_dir_test + '/COCO_val2014_%.12i.jpg' \
                    % this_image
            elif flag == 'val':
                cur_obj_path_s = OBJ_PATH_train_s \
                    + 'COCO_train2014_%.12i.json' % this_image
                image_dir_s = image_dir_val + '/COCO_train2014_%.12i.jpg' \
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
            pair_index = 0
            person_idx = 0
            object_idx = -1
            for person_box in person_bbx:
                for object_box in obj_bbx:
                    ground_truth_this_sample = \
                        ground_truth_this_batch[start_index]
                    score_HOI_this_sample = \
                        score_HOI_this_batch[start_index]

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
                    input_img = img

                    print('''Predictions (Five Highest Confidence Class):{}'''.format(prediction))

                    # Visualize class activation map
                    class_index = predict.argmax(dim=1)
                    class_index = class_index.cpu().numpy()
                    print(ID2VERB[class_index[pair_index]])
                    # predict[pair_index,class_index[pair_index]].backward(retain_graph=True)
                    predict[pair_index,26].backward(retain_graph=True)
                    gradients = model.module.get_activations_gradient()
                    pooled_gradients = torch.mean(gradients, dim=[0,2,3])
                    image_converter = data_loader.Rescale((400, 400))
                    input_img = image_converter(input_img)
                    input_img = input_img.transpose((2,0,1))
                    input_img = input_img.reshape(1,3,400,400)
                    activations = model.module.get_activations(torch.from_numpy(input_img).float().cuda()).detach()

                    for i in range(1024):
                        activations[:,i,:,:] *= pooled_gradients[i]
                    heatmap = torch.mean(activations, dim=1).squeeze()
                    heatmap = np.maximum(heatmap.cpu(), 0)
                    heatmap /= torch.max(heatmap)
                    
                    heatmap = cv2.resize(np.float32(heatmap), (img.shape[1], img.shape[0]))
                    heatmap = np.uint8(255*heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    result_img = cv2.add(heatmap, img)

                    (x, y, w, h) = (int(person_box[0]), int(person_box[1]),
                                    int(person_box[2] - person_box[0]),
                                    int(person_box[3] - person_box[1]))
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0),3)
                    (px, py, pw, ph) = (x,y,w,h)
                    (x, y, w, h) = (int(object_box[0]), int(object_box[1]),
                                    int(object_box[2] - object_box[0]),
                                    int(object_box[3] - object_box[1]))
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255),3)
                    (ox, oy, ow, oh) = (x,y,w,h)
                    print(pw,ph, px, py)
                    # Attention activation map
                    gradients_sp = model.module.get_activations_gradient_sp()
                    pooled_gradients_sp = torch.mean(gradients_sp, dim=[0,2,3])
                    activations_sp = model.module.get_activations_sp(union_box).detach()
                    activations_sp = activations_sp[pair_index].view(1, 32, 13, 13)

                    for i in range(32):
                        activations_sp[:,i,:,:] *= pooled_gradients_sp[i]
                    heatmap_sp = torch.mean(activations_sp, dim=1).squeeze()
                    heatmap_sp = np.maximum(heatmap_sp.cpu(), 0)
                    heatmap_sp /= torch.max(heatmap_sp)
                    
                    heatmap_sp = cv2.resize(np.float32(heatmap_sp), (img.shape[1], img.shape[0]))
                    heatmap_sp = np.uint8(255*heatmap_sp)
                    heatmap_sp = cv2.applyColorMap(heatmap_sp, cv2.COLORMAP_JET)
                    result_img_sp = cv2.add(heatmap_sp, img)

                    # People activation map
                    gradients_people = model.module.get_activations_gradient_people()
                    pooled_gradients_people = torch.mean(gradients_people, dim=[0,2,3])
                    activations_people = model.module.get_activations_people(rois_people).detach()
                    activations_people = activations_people[person_idx].view(1, 1024, 10, 10)

                    for i in range(1024):
                        activations_people[:,i,:,:] *= pooled_gradients_people[i]
                    heatmap_people = torch.mean(activations_people, dim=1).squeeze()
                    heatmap_people = np.maximum(heatmap_people.cpu(), 0)
                    heatmap_people /= torch.max(heatmap_people)
                    
                    heatmap_people = cv2.resize(np.float32(heatmap_people), (pw, ph))
                    nullmap_people = np.zeros((img.shape[0], img.shape[1]))
                    nullmap_people[py:py+ph, px:px+pw] = heatmap_people
                    heatmap_people = np.uint8(255*nullmap_people)
                    heatmap_people = cv2.applyColorMap(heatmap_people, cv2.COLORMAP_JET)
                    result_img_people = cv2.add(heatmap_people, img)

                    # Object activation map
                    if object_idx > -1:
                        gradients_object = model.module.get_activations_gradient_object()
                        pooled_gradients_object = torch.mean(gradients_object, dim=[0,2,3])
                        activations_object = model.module.get_activations_object(rois_object).detach()
                        activations_object = activations_object[object_idx].view(1, 1024, 10, 10)

                        for i in range(1024):
                            activations_object[:,i,:,:] *= pooled_gradients_object[i]
                        heatmap_object = torch.mean(activations_object, dim=1).squeeze()
                        heatmap_object = np.maximum(heatmap_object.cpu(), 0)
                        heatmap_object /= torch.max(heatmap_object)

                        heatmap_object = cv2.resize(np.float32(heatmap_object), (ow, oh))
                        nullmap_object = np.zeros((img.shape[0], img.shape[1]))
                        nullmap_object[oy:oy+oh, ox:ox+ow] = heatmap_object
                        heatmap_object = np.uint8(255*nullmap_object)
                        heatmap_object = cv2.applyColorMap(heatmap_object, cv2.COLORMAP_JET)
                        result_img_object = cv2.add(heatmap_object, img)

                    # Context activation map
                    gradients_context = model.module.get_activations_gradient_context()
                    pooled_gradients_context = torch.mean(gradients_context, dim=[0,2,3])
                    activations_context = model.module.get_activations_context(context).detach()

                    for i in range(1024):
                        activations_context[:,i,:,:] *= pooled_gradients_context[i]
                    heatmap_context = torch.mean(activations_context, dim=1).squeeze()
                    heatmap_context = np.maximum(heatmap_context.cpu(), 0)
                    heatmap_context /= torch.max(heatmap_context)
                    
                    heatmap_context = cv2.resize(np.float32(heatmap_context), (img.shape[1], img.shape[0]))
                    heatmap_context = np.uint8(255*heatmap_context)
                    heatmap_context = cv2.applyColorMap(heatmap_context, cv2.COLORMAP_JET)
                    result_img_context = cv2.add(heatmap_context, img)

                    # Create opencv window
                    img_concate = np.concatenate((result_img, result_img_sp, ), axis=1)
                    img_concate_tmp = np.concatenate((result_img_people, result_img_context), axis=1)
                    img_concate = np.concatenate((img_concate, img_concate_tmp), axis=0)

                    if object_idx > -1:
                        tmp = np.concatenate((np.uint8(np.zeros(result_img_object.shape)), result_img_object), axis=0)
                        img_concate = np.concatenate((img_concate, tmp), axis=1)
                    cv2.namedWindow('heatmap', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('heatmap', int(img_concate.shape[1]*0.7), int(img_concate.shape[0]*0.7))
                    cv2.imshow('heatmap', img_concate)
                    start_index += 1
                    pair_index += 1
                    object_idx += 1 
                    k = cv2.waitKey(0)
                    if k == 27:  # wait for ESC key to exit
                        cv2.destroyAllWindows()
                person_idx += 1
                object_idx = -1
                if k == 27:  # wait for ESC key to exit
                    cv2.destroyAllWindows()
            if k == 27:  # wait for ESC key to exit
                cv2.destroyAllWindows()
                
        elif single_image == 'f' :
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
                    + 'COCO_train2014_%.12i.json' % this_image

                image_dir_s = image_dir_train + '/COCO_train2014_%.12i.jpg' \
                    % this_image
            elif flag == 'test':

                cur_obj_path_s = OBJ_PATH_test_s \
                    + 'COCO_val2014_%.12i.json' % this_image
                image_dir_s = image_dir_test + '/COCO_val2014_%.12i.jpg' \
                    % this_image
            elif flag == 'val':
                cur_obj_path_s = OBJ_PATH_train_s \
                    + 'COCO_train2014_%.12i.json' % this_image
                image_dir_s = image_dir_val + '/COCO_train2014_%.12i.jpg' \
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

                    print('''Predictions (Five Highest Confidence Class):{}'''.format(prediction))

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
