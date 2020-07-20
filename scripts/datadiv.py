import numpy as np
import os
import json
import argparse
import shutil

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

image_id = []
for i in range(29):
    v_list = []
    image_id.append(v_list)
    os.makedirs('../separated_data/train/%s' % list(VERB2ID.keys())[i], exist_ok=True)
    os.makedirs('../separated_data/test/%s' % list(VERB2ID.keys())[i], exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    '--phase',
    type=str,
    required=False,
    default='test',
    help='Choose which phase(test, train, val) you want to divide.'
)

args = parser.parse_args()
data_path = "../All_data/Annotations_vcoco/"
data_phase = args.phase
img_path = '../All_data/Data_vcoco/'


with open(data_path+data_phase+'_annotations.json') as json_file:
    json_data = json.load(json_file)

for d in json_data:
    for v in json_data['{}'.format(d)]:
        if d not in image_id[VERB2ID[v['Verbs']]]:
            image_id[VERB2ID[v['Verbs']]].append(d) 

for (i, idx) in enumerate(image_id):
    for ID in idx:
        if data_phase == 'test':
            shutil.copy(img_path + 'val2014/COCO_val2014_%.12i.jpg' % (int(ID)),
            '../separated_data/test/%s/' % (list(VERB2ID.keys())[i]))
        elif data_phase == 'train':
            shutil.copy(img_path + 'train2014/COCO_train2014_%.12i.jpg' % (int(ID)),
            '../separated_data/train/%s/' % (list(VERB2ID.keys())[i]))
        elif data_phase == 'val':
            shutil.copy(img_path + 'train2014/COCO_train2014_%.12i.jpg' % (int(ID)),
            '../separated_data/train/%s/' % (list(VERB2ID.keys())[i]))
