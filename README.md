# ContextAugment
Implementation of Context-aware methods for multi-label classification to increase AP of minority co-occurrence pair. <br> 
Sogang Univ. Grad.

## Requirements
```shell
Python>=3.8
Pytorch>=1.10.0
pip install -r requirements.txt
```

## Example of co-occurence matrix 

### PascalVOC 2012
<img width="60%" src="https://user-images.githubusercontent.com/81060548/196676003-8572462f-b726-4254-92a2-86694d7ba197.png"/>

### MSCOCO labels : 0 ~ 10
<img width="60%" src="https://user-images.githubusercontent.com/81060548/204138726-2715ef1a-79f5-4e68-8def-28e6a1aeb268.png"/>

# Proposed Methods

## Augmentation for minority
### Method 1 - using a bounding box or polygon segmentation
<img width="30%" src="https://user-images.githubusercontent.com/81060548/196677700-6043b838-4b98-44f1-b88a-6770d684f0c5.png"/>

### Method 2 - image to patch level
<img width="30%" src="https://user-images.githubusercontent.com/81060548/196677869-f73b9882-e4da-4509-9b0c-8d80e424993b.png"/>


# Project Architecture
```shell
├─augmentation.py         # method1, method2, zero co-occurence pair 를 위한 함수 
├─dataset.py              
├─main.py
├─models.py              
├─requirements.txt       
├─utils.py                   
└─README.md
```

## dataloader.py & augmentation.py
- Implementation of augmentation for minority co-occurrence pair(k-pair) from dataset
- Augmentation Method 2(patch-level; similar as Cutmix) 

## Dataset Overview
- Train a model on MS-COCO 2017 train/val dataset
- Test a trained model using MS-COCO 2017 test dataset

- To train a model, we use Pascal-VOC 2012 trainval dataset
- To test a model, we use Pascal-VOC 2007 test dataset

## Model Setting
- We leverage Resnet-18, Resnet-50, Resnet-101 to test our augmentation method.
- Two major options for model, which are from scratch and from pre-trained weights on ImageNet.
- Our purpose of this augmentation is raising APs of minority class. It means that the model should be measured by classes-by-class AP tomeasure our augmentation method properly.

## How to Use
- If you want to apply the original setting, just use train.sh as a script
- To train model with another option, make your own script modifying the hyperparameters.
```
python3 main.py --lr 0.001 \
                --batch-size 64 \
                --scheduler cosine \
                --criterion soft \
                --device cuda:0 \
                --use-method \
                --rotate \
                --save-dir runs/method
```
## Pretrained-weights
```
pip install gdown
gdown https://drive.google.com/uc?id=1wmSHTQnZbqFgaMIg-t-Ga3eocST5QVod # ResNet-101, K=30
```

## Results

### Pascal-VOC
class | mAP | #0 | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11 | #12 | #13 | #14 | #15 | #16 | #17 | #18 | #19 
--- | --- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- |--- | --- | --- |--- |--- |--- |--- |--- |--- 
use-method |  86.24  | 91.96 | 94.7 | 90.65 | 88.31 | 66.1 | 84.85 | 95.14 | 96.6 | 74.97 | 75.66 | 81.75 | 96.6 | 95.97 | 92.54 | 97.57 | 72.41 | 65.46 | 79.93 | 99.45 | 84.24 |---
no-method | 86.16 | 92.07 | 95.65 | 90. | 88.28 | 67.45 | 84.27 | 95.04 | 97.16 | 75. | 73.75 | 82.03 | 97.08 | 95.9 | 91.95 | 97.44 | 70.83 | 65.35 | 81.06 | 98.53 | 84.39 | 

### MS-COCO
class | mAP |
--- | --- | 
no-method |76.46| 
use-method |76.31| 

## ToDo
- [X] To test properly, AP score matrix for multi-label classification
- [X] MS-COCO dataloader with our augmentation method
- [X] Apply K-select option with initial co-occurrence matrix
- [X] Poisition randomness of attached object
- [X] Size randomness of attached object
- [X] Angle randomness of attached object
- [ ] Compare unrel_matrix and performances [evaluate heuristically]
