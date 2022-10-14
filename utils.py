import numbers
import random
import warnings

import torch
from torch import Tensor

import os
import math
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
import pandas as pd

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']


def get_categories(labels_dir):
    """
    Get the object categories
    
    Args:
        label_dir: Directory that contains object specific label as .txt files
    Raises:
        FileNotFoundError: If the label directory does not exist
    Returns:
        Object categories as a list
    """
    
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError
    
    else:
        categories = []
        
        for file in os.listdir(labels_dir):
            if file.endswith("_train.txt"):
                categories.append(file.split("_")[0])
        
        return categories


def encode_labels(target):
    """
    Encode multiple labels using 1/0 encoding 
    
    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """
    
    ls = target['annotation']['object']
  
    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))
  
    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))
    
    k = np.zeros(len(object_categories))
    k[j] = 1
  
    return torch.from_numpy(k)


def get_nrows(file_name):
    """
    Get the number of rows of a csv file
    
    Args:
        file_path: path of the csv file
    Raises:
        FileNotFoundError: If the csv file does not exist
    Returns:
        number of rows
    """
    
    if not os.path.isfile(file_name):
        raise FileNotFoundError
    
    s = 0
    with open(file_name) as f:
        s = sum(1 for line in f)
    return s


def get_mean_and_std(dataloader):
    """
    Get the mean and std of a 3-channel image dataset 
    
    Args:
        dataloader: pytorch dataloader
    Returns:
        mean and std of the dataset
    """
    mean = []
    std = []
    
    total = 0
    r_running, g_running, b_running = 0, 0, 0
    r2_running, g2_running, b2_running = 0, 0, 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            r, g, b = data[:,0 ,:, :], data[:, 1, :, :], data[:, 2, :, :]
            r2, g2, b2 = r**2, g**2, b**2
            
            # Sum up values to find mean
            r_running += r.sum().item()
            g_running += g.sum().item()
            b_running += b.sum().item()
            
            # Sum up squared values to find standard deviation
            r2_running += r2.sum().item()
            g2_running += g2.sum().item()
            b2_running += b2.sum().item()
            
            total += data.size(0)*data.size(2)*data.size(3)
    
    # Append the mean values 
    mean.extend([r_running/total, 
                 g_running/total, 
                 b_running/total])
    
    # Calculate standard deviation and append
    std.extend([
            math.sqrt((r2_running/total) - mean[0]**2),
            math.sqrt((g2_running/total) - mean[1]**2),
            math.sqrt((b2_running/total) - mean[2]**2)
            ])
    
    return mean, std


def plot_history(train_hist, val_hist, y_label, filename, labels=["train", "validation"]):
    """
    Plot training and validation history
    
    Args:
        train_hist: numpy array consisting of train history values (loss/ accuracy metrics)
        valid_hist: numpy array consisting of validation history values (loss/ accuracy metrics)
        y_label: label for y_axis
        filename: filename to store the resulting plot
        labels: legend for the plot
        
    Returns:
        None
    """
    # Plot loss and accuracy
    xi = [i for i in range(0, len(train_hist), 2)]
    plt.plot(train_hist, label = labels[0])
    plt.plot(val_hist, label = labels[1])
    plt.xticks(xi)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.show()


def get_ap_score(y_true, y_scores):
    """
    Get average precision score between 2 1-d numpy arrays
    
    Args:
        y_true: batch of true labels
        y_scores: batch of confidence scores
=
    Returns:
        sum of batch average precision
    """
    scores = 0.0
    
    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true = y_true[i], y_score = y_scores[i])
    
    return scores

def save_results(images, scores, columns, filename):
    """
    Save inference results as csv
    
    Args:
        images: inferred image list
        scores: confidence score for inferred images
        columns: object categories
        filename: name and location to save resulting csv
    """
    df_scores = pd.DataFrame(scores, columns=columns)
    df_scores['image'] = images
    df_scores.set_index('image', inplace=True)
    df_scores.to_csv(filename)


def append_gt(gt_csv_path, scores_csv_path, store_filename):
    """
    Append ground truth to confidence score csv
    
    Args:
        gt_csv_path: Ground truth csv location
        scores_csv_path: Confidence scores csv path
        store_filename: name and location to save resulting csv
    """
    gt_df = pd.read_csv(gt_csv_path)
    scores_df = pd.read_csv(scores_csv_path)
    
    gt_label_list = []
    for index, row in gt_df.iterrows():
        arr = np.array(gt_df.iloc[index,1:], dtype=int)
        target_idx = np.ravel(np.where(arr == 1))
        j = [object_categories[i] for i in target_idx]
        gt_label_list.append(j)
    
    scores_df.insert(1, "gt", gt_label_list)
    scores_df.to_csv(store_filename, index=False)

        

def get_classification_accuracy(gt_csv_path, scores_csv_path, store_filename):
    """
    Plot mean tail accuracy across all classes for threshold values
    
    Args:
        gt_csv_path: Ground truth csv location
        scores_csv_path: Confidence scores csv path
        store_filename: name and location to save resulting plot
    """
    gt_df = pd.read_csv(gt_csv_path)
    scores_df = pd.read_csv(scores_csv_path)
    
    # Get the top-50 images
    top_num = 2800
    image_num = 2
    num_threshold = 10
    results = []
    
    for image_num in range(1, 21):
        clf = np.sort(np.array(scores_df.iloc[:,image_num], dtype=float))[-top_num:]
        ls = np.linspace(0.0, 1.0, num=num_threshold)
        
        class_results = []
        for i in ls:
            clf = np.sort(np.array(scores_df.iloc[:,image_num], dtype=float))[-top_num:]
            clf_ind = np.argsort(np.array(scores_df.iloc[:,image_num], dtype=float))[-top_num:]
            
            # Read ground truth
            gt = np.sort(np.array(gt_df.iloc[:,image_num], dtype=int))
            
            # Now get the ground truth corresponding to top-50 scores
            gt = gt[clf_ind]
            clf[clf >= i] = 1
            clf[clf < i] = 0
            
            score = accuracy_score(y_true=gt, y_pred=clf, normalize=False)/clf.shape[0]
            class_results.append(score)
        
        results.append(class_results)
    
    results = np.asarray(results)
    
    ls = np.linspace(0.0, 1.0, num=num_threshold)
    plt.plot(ls, results.mean(0))
    plt.title("Mean Tail Accuracy vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Mean Tail Accuracy")
    plt.savefig(store_filename)
    plt.show()
            

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class ContextAugmentation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def make_foreground_instance(img,mask,label):
        """ 
        Make foreground object 
        numpy array로 instance_img를 20개 만들어놓고 forward에서 붙여도 되는지 확인 필요
        """
        mask=np.where(mask[:,:]!=label,0,mask[:,:]) 
        mask=np.where(mask[:,:]!=0,1,mask[:,:])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j]=img[i][j]*mask[i][j]
        return instance_img 

    def forward(self, img, mask, instance_img):

        instance_img = self.make_foreground_instance(img, mask, label)
    
        y_axis=mask.shape[0]//2 # 4분면을 위한 중심 축 y
        x_axis=mask.shape[1]//2 # 4분면을 위한 중심 축 x
        quad_area=np.zeros(4) # 각 4분면의 instance 가 차지하는 영역을 저장하기 위한 array
        quad_img=[] # 각 4분면의 이미지를 저장

        
        # 4분면 슬라이싱을 위한 인덱스 저장
        quad_idx=[((0,y_axis),(x_axis,mask.shape[1])),((0,y_axis),(0,x_axis)),((y_axis,mask.shape[0]),(0,x_axis)),((y_axis,mask.shape[0]),(x_axis,mask.shape[1]))] 
        
        # instasnce가 있는 부분을 1로 바꾼 임시 리스트
        temp=np.where(mask[:,:]!=0,1,mask[:,:])
        
        # 임시 리스트의 값을 합산하여 각 4분면의 instance가 차지하는 면적 구하기
        for i in range(4):
            quad_area[i]=np.sum(temp[quad_idx[i][0][0]:quad_idx[i][0][1],quad_idx[i][1][0]:quad_idx[i][1][1]])
            # 4분면 이미지 저장
            quad_img.append(img[quad_idx[i][0][0]:quad_idx[i][0][1],quad_idx[i][1][0]:quad_idx[i][1][1],:])
        
        #가장 적은 면적 차지하는 4분면 선택
        selected_quad=np.argmin(quad_area)
        
        # 선택된 4분면에 따른 resize 크기 추출 ( 절반으로 고정할 경우 홀수일 때 문제 발생)
        resize_y,resize_x,_=quad_img[selected_quad].shape
        
        # resize크기에 따른 instance img resize
        resized_instance=np.array(Image.fromarray(instance_img).resize((resize_x,resize_y)))
        
        # resize한 instance 이미지 선택된 4분면의 이미지에 합성
        quad_img[selected_quad]=np.where(resized_instance[:,:,:]!=0,resized_instance[:,:,:],quad_img[selected_quad][:,:,:])
        
        #이미지 복원
        temp_img_1=np.concatenate((quad_img[1],quad_img[0]),axis=1)
        temp_img_2=np.concatenate((quad_img[2],quad_img[3]),axis=1)
        recon_img=np.concatenate((temp_img_1,temp_img_2),axis=0)
        
        return recon_img
    