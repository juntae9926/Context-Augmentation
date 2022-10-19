# Context-Augmentation
Implementation of Context considered Data Augmentation. <br>
Sogang Univ. Grad. Big-data project repository

## Co-Occurence in PascalVOC2012
<img width="60%" src="https://user-images.githubusercontent.com/81060548/196676003-8572462f-b726-4254-92a2-86694d7ba197.png"/>

## Proposed Methods
# Method 1 - using a bounding box or polygon segmentation
<img width="40%" src="https://user-images.githubusercontent.com/81060548/196677700-6043b838-4b98-44f1-b88a-6770d684f0c5.png"/>

# Method 2 - image to patch level
<img width="40%" src="https://user-images.githubusercontent.com/81060548/196677869-f73b9882-e4da-4509-9b0c-8d80e424993b.png"/>

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
