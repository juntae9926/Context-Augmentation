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


class PascalVOC:
    def __init__(self, root):
        self.root_dir = root
        self.img_dir =  os.path.join(root, 'JPEGImages/')
        self.ann_dir = os.path.join(root, 'Annotations')
        self.set_dir = os.path.join(root, 'ImageSets', 'Main')
        self.cache_dir = os.path.join(root, 'csvs')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)


class PascalVOC_Dataset(voc.VOCDetection):
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None, use_method1 = False):
        
        super().__init__(
             root="./", 
             year=year, 
             image_set=image_set, 
             download=download, 
             transform=transform, 
             target_transform=target_transform)
        
        self.root = os.path.join(root, "VOC" + year)
        print(self.root)
        self.label_img, _ = self.initialize_dict()
        self.use_method1 = use_method1

        self.unrel_pairs = self.get_unrel_pairs()
        self.segments_buffer = self.get_stitch_segments()
        self.bboxes_buffer = self.get_stitch_bboxes()

        _, self.segments_num_dict = self.initialize_dict()
        _, self.bboxes_num_dict = self.initialize_dict()
        _, self.method2_num_dict = self.initialize_dict()

    
    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(elemTree.parse(self.annotations[index]).getroot())

        # try method1 on image
        if self.use_method1:
            _, test_target = self.transforms(img, target)
            is_segmented = False if target['annotation']['segmented'] == str(0) else True
            target_list = list(np.where(test_target == 1)[0])
            current_single_label = random.choice(target_list)
            try:
                aug_label = random.choice(np.where(self.unrel_pairs[current_single_label] == 1)[0])
                pair = [current_single_label, aug_label]
                img = self.method1(pair, img, index, is_segmented)
            except:
                pair = None

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # try method1 on label
        if self.use_method1 and pair:
            target[aug_label] = 1
    
        return img, target
    

    def collate_fn(batch):
        return batch
        
    
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
                    unrel_pairs.append((i, j))

        numpy_unrel_pairs = np.zeros((20, 20), dtype=int)
        for i, j in unrel_pairs:
            numpy_unrel_pairs[i, j] = 1

        return numpy_unrel_pairs


    def get_stitch_segments(self, num_instances=5):
        segments = dict()
        for label in range(len(labels)):
            bg = [self.label_img[str(label)][i].split('.')[0] for i in range(num_instances)]

            temp = []
            for file_name in bg:
                instance = np.array(Image.open(os.path.join(self.root, "SegmentationClass",  file_name + ".png")))
                temp.append(instance)

            segments[str(label)] = temp
        
        return segments
    
    def get_stitch_bboxes(self, num_instances=5):
        bboxes = dict()
        for label in range(len(labels)):
            bg = [self.label_img[str(label)][i].split('.')[0] for i in range(num_instances)]

            temp = []
            for file_name in bg:
                with open(os.path.join(self.root, "Annotations", file_name + ".xml")) as fd:
                    annotations = xmltodict.parse(fd.read())['annotation']['object']

                    img = np.array(Image.open(os.path.join(self.root, "JPEGImages",  file_name + ".jpg")))

                    if type(annotations) == list:
                        for i in annotations:
                            if labels.index(i['name']) == label:
                                y1, y2, x1, x2 = int(i['bndbox']['ymin']), int(i['bndbox']['ymax']), int(i['bndbox']['xmin']), int(i['bndbox']['xmax'])
                    else:
                        i = annotations
                        if labels.index(i['name']) == label:
                            y1, y2, x1, x2 = int(i['bndbox']['ymin']), int(i['bndbox']['ymax']), int(i['bndbox']['xmin']), int(i['bndbox']['xmax'])
                    mask = img[y1:y2, x1:x2, :]
                    temp.append(mask)
            
            bboxes[str(label)] = temp
        
        return bboxes

    def _make_mask(self, img, file_name, instance_label):
        with open(os.path.join(self.root, "Annotations", file_name + ".xml")) as fd:
            label = xmltodict.parse(fd.read())['annotation']['object']
        bbox=[]
        labels=[]

        if isinstance(label,list):
            labels=label[:]
        else:
            labels.append(label)

        for i in labels:
            bbox.append([i['bndbox']['ymin'],i['bndbox']['ymax'],i['bndbox']['xmin'],i['bndbox']['xmax']])

        for i in range(len(bbox)):
            bbox[i]=list(map(int,bbox[i]))
        
        mask=np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
        for i in bbox:
            x_size,y_size=mask[i[0]:i[1],i[2]:i[3]].shape

            mask[i[0]:i[1],i[2]:i[3]]=np.full([x_size,y_size], instance_label)
            
        return mask

    def _make_mixed_image(self, img, mask, img2):
        y_axis = mask.shape[0]//2 # 4분면을 위한 중심 축 y
        x_axis = mask.shape[1]//2 # 4분면을 위한 중심 축 x
        quad_area = np.zeros(4) # 각 4분면의 instance가 차지하는 영역을 저장하기 위한 array
        quad_img = [] # 각 4분면의 이미지를 저장
        
        # 4분면 슬라이싱을 위한 인덱스 저장
        quad_idx=[((0, y_axis),(x_axis, mask.shape[1])),
                ((0, y_axis),(0,x_axis)),
                ((y_axis, mask.shape[0]),(0, x_axis)),
                ((y_axis, mask.shape[0]),(x_axis, mask.shape[1]))]
        
        # instance가 있는 부분을 1로 바꾼 임시 리스트
        temp = np.where(mask[:,:]!=0, 1, mask[:, :])
        
        # 임시 리스트의 값을 합산하여 각 4분면의 instance가 차지하는 면적 구하기
        for i in range(4):
            quad_area[i] = np.sum(temp[quad_idx[i][0][0]:quad_idx[i][0][1],quad_idx[i][1][0]:quad_idx[i][1][1]])
            # 4분면 이미지 저장
            quad_img.append(img[quad_idx[i][0][0]:quad_idx[i][0][1],quad_idx[i][1][0]:quad_idx[i][1][1],:])
            
        # 가장 적은 면적 차지하는 4분면 선택
        selected_quad = np.argmin(quad_area)
        
        # 선택된 4분면에 따른 resize 크기 추출
        resize_y, resize_x, _ = quad_img[selected_quad].shape
        
        # resize 크기에 따른 img resize
        resize_patch = np.array(Image.fromarray(img2).resize((resize_x, resize_y)))
        
        # resize한 image를 선택된 4분면 이미지에 합성
        quad_img[selected_quad] = np.where(resize_patch[:,:,:]!=0, resize_patch[:,:,:], quad_img[selected_quad][:,:,:])
        
        # 이미지 복원
        temp_img_1 = np.concatenate((quad_img[1], quad_img[0]), axis = 1)
        temp_img_2 = np.concatenate((quad_img[2], quad_img[3]), axis = 1)
        recon_img = np.concatenate((temp_img_1, temp_img_2), axis = 0)
        
        # 복원된 합성 이미지 반환
        return recon_img

    
    def method1(self, pair, img, index, is_segmented):
        file_name = self.images[index].split('/')[-1].split('.')[0]
        bg_img = np.array(img)
        if is_segmented: 
            bg_img_mask = np.array(Image.open(os.path.join('./', self.root, "SegmentationClass", file_name + ".png")))
            segment = self.segments_buffer[str(pair[1])][self.segments_num_dict[str(pair[1])]] # Img: np.array
            self.segments_num_dict[str(pair[1])] += 1
            mixed_img = self._make_mixed_image(bg_img, bg_img_mask, segment)
        else:
            bg_img_mask = self._make_mask(bg_img, file_name, pair[0]+1)
            bbox = self.bboxes_buffer[str(pair[1])][self.bboxes_num_dict[str(pair[1])]] # Img: np.array
            self.bboxes_num_dict[str(pair[1])] += 1
            mixed_img = self._make_mixed_image(bg_img, bg_img_mask, bbox)

        return Image.fromarray(mixed_img)


    def method2(self, pair, img, index, is_segmented):
        file_name = self.images[index].split('/')[-1].split('.')[0]
        bg_img = np.array(img)

        # if is_segmented: 
        #     bg_img_mask = np.array(Image.open(os.path.join('./', self.root, "SegmentationClass", file_name + ".png")))
        #     segment = self.segments_buffer[str(pair[1])][self.segments_num_dict[str(pair[1])]] # Img: np.array
        #     mixed_img = self._make_mixed_image(bg_img, bg_img_mask, segment)
        # else:
        #     bg_img_mask = self._make_mask(bg_img, file_name, pair[0]+1)
        #     bbox = self.bboxes_buffer[str(pair[1])][self.bboxes_num_dict[str(pair[1])]] # Img: np.array
        #     mixed_img = self._make_mixed_image(bg_img, bg_img_mask, bbox)

        # self.method2_num_dict += 1

        # return Image.fromarray(mixed_img)