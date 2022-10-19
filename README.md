# Context-Augmentation
Sogang Univ. Grad. Big-data project repository

## Co-Occurence in PascalVOC2012
<img width="80%" src="https://user-images.githubusercontent.com/81060548/196676003-8572462f-b726-4254-92a2-86694d7ba197.png"/>

## Project Architecture
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
