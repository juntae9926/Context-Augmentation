import xml.etree.ElementTree as elemTree
import os
import numpy as np


def read_annotation(annot_path):
    for annot in os.listdir(annot_path):
        tree = elemTree.parse(os.path.join(annot_path, annot))
        obj_list = tree.findall('./object')
        for item in obj_list:
            obj = item.find('./name').text


labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
          'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
          'train', 'tvmonitor']
path = os.path.join(os.getcwd(), 'data/Annotations')
read_annotation(path)
