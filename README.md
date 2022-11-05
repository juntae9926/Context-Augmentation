# ContextAugment
Implementation of Context-aware methods for multi-label classification. <br> 
Sogang Univ. Grad.

## Example of co-occurence matrix (PascalVOC 2012)
<img width="60%" src="https://user-images.githubusercontent.com/81060548/196676003-8572462f-b726-4254-92a2-86694d7ba197.png"/>

# Proposed Methods

## Augmentation for minority
### Method 1 - using a bounding box or polygon segmentation
<img width="30%" src="https://user-images.githubusercontent.com/81060548/196677700-6043b838-4b98-44f1-b88a-6770d684f0c5.png"/>

### Method 2 - image to patch level
<img width="30%" src="https://user-images.githubusercontent.com/81060548/196677869-f73b9882-e4da-4509-9b0c-8d80e424993b.png"/>

## Semi-supervised method for majority
- MoCov2


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

## augmentation.py
- Pascal VOC 2012에서 co-occurance가 0인 category pair에서 랜덤으로 이미지를 선택해서 1장씩 augmentation하는 로직까지 완성
- patch 단위로 붙히는 method 2 완성

## Dataset Overview
- To train a CNN model, we use VOC2012 trainval dataset
- To test a CNN model, we use VOC2007 test dataset

## Model Setting
- We leverage Resnet-18, Resnet-50, Resnet-101 to test our augmentation method.
- Two major options for model, which are from scratch and from pre-trained weights on ImageNet.
- To measure our augmentation method properly, It needs to be measured by class-by-class mAP.

## How to Use
- If you want to apply the original setting, just use train.sh as a script
- For modifying other options, you can make a script as below.
```
python3 main.py --lr 0.001 \
                --batch-size 64 \
                --scheduler cosine \
                --criterion soft \
                --device cuda:0 \
                --method1 \
                --save-dir runs/method1
```

## Results


class | mAP | #0 | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11 | #12 | #13 | #14 | #15 | #16 | #17 | #18 | #19 
--- | --- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- |--- | --- | --- |--- |--- |--- |--- |--- |--- 
no-method |  86.24  | 91.96 | 94.7 | 90.65 | 88.31 | 66.1 | 84.85 | 95.14 | 96.6 | 74.97 | 75.66 | 81.75 | 96.6 | 95.97 | 92.54 | 97.57 | 72.41 | 65.46 | 79.93 | 99.45 | 84.24 |---
method1 | 86.16 | 92.07 | 95.65 | 90. | 88.28 | 67.45 | 84.27 | 95.04 | 97.16 | 75. | 73.75 | 82.03 | 97.08 | 95.9 | 91.95 | 97.44 | 70.83 | 65.35 | 81.06 | 98.53 | 84.39 | 



## ToDo
- [X] Implement AP score matrix for multi-label classification
- [ ] Apply minimum selection algorithm on datalaader
- [ ] Visualize co-occurrence matrix every epoch [main - epoch]
- [ ] Compare unrel_matrix and performances [evaluate heuristically]
- [ ] Using MOCOv2 to prevent performance of majority pair 
