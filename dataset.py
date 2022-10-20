import torchvision.datasets.voc as voc
import xml.etree.ElementTree as elemTree
from typing import Any, Dict

import collections
import numpy as np
from itertools import permutations, combinations

from augmentation import *

labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
          'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
          'train', 'tvmonitor']

# Segmentation ONLY
class PascalVOC_Dataset(voc.VOCDetection):
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        
        super().__init__(
             root="./", 
             year=year, 
             image_set=image_set, 
             download=download, 
             transform=transform, 
             target_transform=target_transform)
        
        # self.dataset_path = dataset_path
        self.root = root
        self.label_img, _ = self.initialize_dict()

        self.unrel_pairs = self.get_unrel_pairs()
        self.segments_buffer = self.get_segments()

        _, self.num_dict = self.initialize_dict()

    
    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(elemTree.parse(self.annotations[index]).getroot())

        _, test_target = self.transforms(img, target)

        target_list = list(np.where(test_target == 1)[0])
        if len(target_list) > 1:
            for pair in list(combinations(target_list, 2)): 
                pair = (18, 19)
                if list(pair) in self.unrel_pairs:
                    img, target = self.method1(pair, img, target, index)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
        
    
    def __len__(self):
        return len(self.images)


    def initialize_dict(self):
        label_dict = dict()
        num_dict = dict()
        for label in range(len(labels)):
            label_dict.setdefault(str(label), [])
            num_dict.setdefault(str(label), 0)
        
        return label_dict, num_dict


    @staticmethod
    def parse_voc_xml(node: elemTree) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(PascalVOC_Dataset.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text

        return voc_dict
    

    def _get_comatrix(self):
        annot_path = os.path.join(self.root, "Annotations")
        co_matrix = np.zeros([len(labels), len(labels)], dtype=int)
        for annot in os.listdir(annot_path):

            tree = elemTree.parse(os.path.join(annot_path, annot))
            obj_list = tree.findall('./object')
            is_segmented = tree.find('./segmented').text
            if is_segmented == '0':
                continue
            occur_list = []
            for item in obj_list:
                obj = item.find('./name').text
                if obj not in occur_list:
                    occur_list.append(obj)
                    self.label_img[str(labels.index(obj))].append(annot)
            for occ_i in occur_list:
                for occ_j in occur_list:
                    co_matrix[labels.index(occ_i)][labels.index(occ_j)] += 1

        return co_matrix
    

    def get_unrel_pairs(self):
        co_matrix = self._get_comatrix()
        unrel_pairs = []
        for i in range(20):
            for j in range(i, 20):
                if co_matrix[i][j] == 0:
                    unrel_pairs.append([i, j])

        return unrel_pairs


    def get_segments(self, num_instances=5):
        segments = dict()
        for label in range(len(labels)):
            bg = [self.label_img[str(label)][i].split('.')[0] for i in range(num_instances)]

            temp = []
            for file_name in bg:
                instance = np.array(Image.open(os.path.join(self.root, "SegmentationClass",  file_name + ".png")))
                temp.append(instance)

            segments[str(label)] = temp
        
        return segments
    

    def method1(self, pair, img, target, index):
        filename = self.images[index].split('/')[-1].split('.')[0]
        bg_img = np.array(img)
        # try: 
        #     bg_img_mask = np.array(Image.open(os.path.join('./', self.root, "SegmentationClass", filename + ".png")))

        #     segment = self.segments_buffer[str(pair[1])][self.num_dict[str(pair[1])]] # Img: np.array
        #     self.num_dict[str(pair[1])] += 1 

        #     y_axis = bg_img_mask.shape[0]//2 # 4분면을 위한 중심 축 y
        #     x_axis = bg_img_mask.shape[1]//2 # 4분면을 위한 중심 축 x
        #     quad_area = np.zeros(4) # 각 4분면의 instance가 차지하는 영역을 저장하기 위한 array
        #     quad_img = [] # 각 4분면의 이미지를 저장
            
        #     # 4분면 슬라이싱을 위한 인덱스 저장
        #     quad_idx=[((0, y_axis),(x_axis, bg_img_mask.shape[1])),
        #             ((0, y_axis),(0,x_axis)),
        #             ((y_axis, bg_img_mask.shape[0]),(0, x_axis)),
        #             ((y_axis, bg_img_mask.shape[0]),(x_axis, bg_img_mask.shape[1]))]
            
        #     # instance가 있는 부분을 1로 바꾼 임시 리스트
        #     temp = np.where(bg_img_mask[:,:]!=0, 1, bg_img_mask[:, :])

        #     for i in range(4):
        #         quad_area[i] = np.sum(temp[quad_idx[i][0][0]:quad_idx[i][0][1],quad_idx[i][1][0]:quad_idx[i][1][1]])
        #         # 4분면 이미지 저장
        #         quad_img.append(img[quad_idx[i][0][0]:quad_idx[i][0][1],quad_idx[i][1][0]:quad_idx[i][1][1],:])

        # except:
        #     self.annotations
        return img, target