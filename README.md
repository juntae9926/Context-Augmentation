# ContextAugment
Implementation of Context considered for Data Augmentation. <br>
Repository of Big Data Computing(CSE6528, AIE6202)  
Sogang Univ. Grad.

## Co-Occurence in PascalVOC2012 Dataset
<img width="60%" src="https://user-images.githubusercontent.com/81060548/196676003-8572462f-b726-4254-92a2-86694d7ba197.png"/>

# Proposed Methods

## Method 1 - using a bounding box or polygon segmentation
<img width="30%" src="https://user-images.githubusercontent.com/81060548/196677700-6043b838-4b98-44f1-b88a-6770d684f0c5.png"/>

## Method 2 - image to patch level
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

## Example of resultlog (PascalVOC 2012)
<img width="771" alt="image" src="https://user-images.githubusercontent.com/79918796/198234488-53498698-4eaf-47d4-bbc2-8be8d5e92b4a.png">


## ToDo
- [ ] Apply minimum selection algorithm on datalaader [dataloader - method 1]
- [ ] Visualize co-occurrence matrix every epoch [main - epoch]
- [ ] Compare unrel_matrix and performances [evaluate heuristically]
- [ ] Test pre-trained weights with freezing
