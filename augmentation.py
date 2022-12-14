# from __future__ import annotatiosns
from asyncio.proactor_events import _ProactorBaseWritePipeTransport
import random
import os
import numpy as np
import xml.etree.ElementTree as elemTree
from PIL import Image
from tqdm.notebook import tqdm
import xmltodict
import argparse
from pycocotools.coco import COCO

labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
          'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
          'train', 'tvmonitor']
label_img = {}

# 91 * 91 size co-occurrence matrix
def get_comatrix_coco():
    annFile = 'data/annotations/instances_train2017.json'
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    labels = [cat['name'] for cat in cats]
    co_matrix = np.zeros([92, 92])
    for i in [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:
        co_matrix[i] = np.inf
    for i in range(582000):
        annIds = coco.getAnnIds(imgIds = i)
        catIds = set()
        for annId in annIds:
            ann = coco.loadAnns(annId)
            catIds.add(ann[0]['category_id'])
        for occ_i in catIds:
            for occ_j in catIds:
                co_matrix[occ_i][occ_j] += 1
    return co_matrix


# read files and return co-occurence matrix
def get_comatrix(dataset_path, segmented=False):
    annot_path = os.path.join(dataset_path, "Annotations")
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


# 데이터 읽기
def read_data(dataset_path ,pair ,is_segmented=True):
    temp1 = []
    temp2 = []
    class1,class2 = pair

    path_img1 = os.path.join(dataset_path, "JPEGImages", class1[1]+".jpg")
    path_img2 = os.path.join(dataset_path, "JPEGImages", class2[1]+".jpg")

    path_label1 = os.path.join(dataset_path, "Annotations",class1[1]+".xml")
    path_label2 = os.path.join(dataset_path, "Annotations",class2[1]+".xml")

    if is_segmented:
        path_mask1=os.path.join(dataset_path, "SegmentationClass", class1[1]+".png")
        path_mask2=os.path.join(dataset_path, "SegmentationClass", class2[1]+".png")

    else:
        path_mask1=None
        path_mask2=None
        
    temp1.append([path_img1, path_label1, path_mask1])
    temp2.append([path_img2, path_label2, path_mask2])
        
    return temp1, temp2


def save_numpy_image(file_name, img):
    img = Image.fromarray(img)
    img.save(file_name, 'png')

# method 1에 사용되는 함수
def make_instance(mask, img, label):
    mask=np.where(mask[:,:]!=label,0,mask[:,:]) #mask img를 통해서 원하는 label 제외하고 전부 0으로 변경
    mask=np.where(mask[:,:]!=0,1,mask[:,:])# 잘린 instance 값을 1로 변경하여 곱 연산을 위한 mask로 변경 (1곱해서 그대로 두고 0곱해서 없애기) 
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j]=img[i][j]*mask[i][j] # rgb에 대한 mask 적용

    return img # mask로 추출된 instance image 추출( 원하는 부분만 원래 값을 사용하고, 나머지는 0인 이미지 )

def make_mask_poly(img,segment_anno,instance_label):

    masks=[]
    for i in segment_anno:

        mask_img=Image.new('L',(img.shape[1],img.shape[0]),0)
        ImageDraw.Draw(mask_img).polygon(i,outline=instance_label,fill=instance_label)
        masks.append(numpy.array(mask_img))
    
    result=np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
    for i in masks:
        result+=i

    return result

# methode 1에 사용되는 함수
def make_mask(img,label,instance_label):
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

        mask[i[0]:i[1],i[2]:i[3]]=np.full([x_size,y_size],instance_label)
        
    return mask


def make_mixed_image(img, mask, img2):
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
    
    # resize 크기에 따른 imag resize
    resize_patch = np.array(Image.fromarray(img2).resize((resize_x, resize_y)))
    
    # resize한 image를 선택된 4분면 이미지에 합성
    quad_img[selected_quad] = np.where(resize_patch[:,:,:]!=0, resize_patch[:,:,:], quad_img[selected_quad][:,:,:])
    
    # 이미지 복원
    temp_img_1 = np.concatenate((quad_img[1], quad_img[0]), axis = 1)
    temp_img_2 = np.concatenate((quad_img[2], quad_img[3]), axis = 1)
    recon_img = np.concatenate((temp_img_1, temp_img_2), axis = 0)
    
    # 복원된 합성 이미지 반환
    return recon_img


def method1(dataset_path, pair, is_segmented=True):
    data1, data2 = read_data(dataset_path, pair=pair,is_segmented=is_segmented)

    img1 = np.array(Image.open(data1[0][0]))
    img2 = np.array(Image.open(data2[0][0]))

    label1=pair[0][0]+1
    label2=pair[1][0]+1

    if is_segmented:
        mask1 = np.array(Image.open(data1[0][2]))
        mask2 = np.array(Image.open(data2[0][2]))
    else:
        with open(data1[0][1]) as fd:
            annotation1=xmltodict.parse(fd.read())
        with open(data2[0][1]) as fd:
            annotation2=xmltodict.parse(fd.read())
        
        mask1=make_mask(img1,annotation1['annotation']['object'],label1)
        mask2=make_mask(img2,annotation1['annotation']['object'],label2)
        
    instance_img=make_instance(mask1,img1,label1)
    mixed_img = make_mixed_image(img2, mask2, instance_img)
    save_numpy_image(f'test.png', mixed_img)

        
def method2(dataset_path):
    path = os.getcwd()
    data1, data2 = read_data(dataset_path, pair=pair, is_segmented=is_segmented)
        
    img1 = np.array(Image.open(data1[0][0]))
    img2 = np.array(Image.open(data2[0][0]))
    
    label1 = pair[0][0]
    label2 = pair[1][0]
    
    if is_segmented:
        mask1 = np.array(Image.open(data1[0][2]))
        mixed_img = make_mixed_image(img1, mask1, img2)
        save_numpy_image(f'segmented_method2.png', mixed_img)
    else:
        with open(data1[0][1], 'r') as reader:
            annotation1 = xmltodict.parse(reader.read())
        mask1 = make_mask(img1, annotation1['annotation']['object'], label1) 
        mixed_img = make_mixed_image(img1, mask1, img2)
        save_numpy_image(f'bbox_method2.png', mixed_img)

def test():
    with open('annotations/result.json','r') as f:
        json_data=json.load(f)

    img_list=[]
    img_name=[]
    cnt=0
    for i in json_data:
        if cnt>=2:
            break

        img = np.array(Image.open('val2017/'+i.zfill(12)+".jpg"))
        img_name.append(i)
        img_list.append(img)
        cnt+=1
    
    mask1=make_mask_poly(img_list[0],json_data[img_name[0]]['segmentation'][1],json_data[img_name[0]]['category_id'][0])
    mask2=make_mask_poly(img_list[1],json_data[img_name[1]]['segmentation'][1],json_data[img_name[1]]['category_id'][0])
    
    instance_img=make_instance(mask1,img_list[0],json_data[img_name[0]]['category_id'][0])
    mixed_img = make_mixed_image(img_list[1], mask2, instance_img)
    save_numpy_image(f'test.png', mixed_img)

        
if __name__ ==  "__main__":
    # test()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="./VOCdevkit/VOC2012", type=str, help="get dataset path")
    args = parser.parse_args()

    initialize_dict()
    # segmented or non segmented data
    is_segmented=True

    co_occur = get_comatrix(args.dataset_path, segmented=is_segmented)
    co_matrix = get_comatrix_coco()
    # print(co_occur)
    unrel_pairs = get_unrel_pairs(co_occur)
    print(f"number of unrelated pairs: {len(unrel_pairs)}")

    # 배경, 인스턴스 pair : [label_idx, filename] list . ex) [[(0, 2007_000032), (1, 2007_000033)], ... ]
    bg_target_pairs = get_bg_target(unrel_pairs)

    for i in bg_target_pairs:
        method1(dataset_path=args.dataset_path, pair=i, is_segmented=is_segmented) # 찾은 pair , segment 여부 input
        method2(dataset_path=args.dataset_path, pair=i, is_segmented=is_segmented)
