import xml.etree.ElementTree as elemTree
import os
import numpy as np
import random

labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
          'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
          'train', 'tvmonitor']
path = os.path.join(os.getcwd(), 'data/Annotations')
label_img = {}

# read files and return co-occurence matrix
def get_comatrix(annot_path, segmented=False):
    co_matrix = np.zeros([len(labels), len(labels)], dtype=int)
    for annot in os.listdir(annot_path):
        tree = elemTree.parse(os.path.join(annot_path, annot))
        obj_list = tree.findall('./object')
        is_segmented = tree.find('./segmented').text
        if segmented and is_segmented == '0':
            continue
        occur_list = []
        for item in obj_list:
            obj = item.find('./name').text
            if obj not in occur_list:
                occur_list.append(obj)
                label_img[obj].append(annot)
        for occ_i in occur_list:
            for occ_j in occur_list:
                co_matrix[labels.index(occ_i)][labels.index(occ_j)] += 1
    return co_matrix


def initialize_dict():
    for label in labels:
        label_img.setdefault(label, [])


def get_bg_target(unrel_list):
    pairs = []
    for unrel_pair in unrel_list:
        bg = random.choice(label_img[labels[unrel_pair[0]]])
        bg = str.split(bg, '.')[0]
        t = random.choice(label_img[labels[unrel_pair[1]]])
        t = str.split(t, '.')[0]
        pairs.append([(unrel_pair[0], bg), (unrel_pair[1], t)])
    return pairs

def get_unrel_pairs(mat):
    '''
    :param mat: [[178, 0, 0, 1, ...],
                 [0, 144, ...],...]
    :return: [[(0, 2007_000032), (1, 2007_000033)], ...]
            [[background, instance_target], ...]
    '''

    ret = []
    zero_occur = np.where(mat == 0)
    for i in range(len(zero_occur[0])):
        ret.append([zero_occur[0][i], zero_occur[1][i]])
    return ret
# def get_img_mask()


initialize_dict()
co_occur = get_comatrix(path, True)
unrel_pairs = get_unrel_pairs(co_occur)

# 배경, 인스턴스 짝들의 list. ex) [[(0, 2007_000032), (1, 2007_000033)], ... ]
bg_target_pairs = get_bg_target(unrel_pairs)

