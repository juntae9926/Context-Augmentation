
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import numpy as np
import math

#데이터 읽기 (VOC 폴더 내부에서 실행 , image, annotation 폴더와 같은 경로)
def Read_Data(path,is_train = True):
    temp = []
    updated_path = os.path.join(path,"ImageSets","Segmentation","train.txt" if is_train else "val.txt")
    with open(updated_path,"r") as file_:
        Instances = file_.read().split()
        for img in Instances:
            path_img = os.path.join(path,"JPEGImages",img+".jpg")
            path_label = os.path.join(path,"SegmentationClass",img+".png")
            temp.append([path_img,path_label])
    return temp

def make_instance(mask,img,label):
    mask=np.where(mask[:,:]!=label,0,mask[:,:]) #mask img를 통해서 원하는 label 제외하고 전부 0으로 변경
    mask=np.where(mask[:,:]!=0,1,mask[:,:])# 잘린 instance 값을 1로 변경하여 곱 연산을 위한 mask로 변경 (1곱해서 그대로 두고 0곱해서 없애기) 
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j]=img[i][j]*mask[i][j] # rgb에 대한 mask 적용
            
    return img # mask로 추출된 instance image 추출( 원하는 부분만 원래 값을 사용하고, 나머지는 0인 이미지 )

def show_numpy_image(img):
    img=Image.fromarray(img)
    plt.imshow(img)
    

def save_numpy_image(file_name,img):
    img=Image.fromarray(img)
    img.save(file_name,'png')
    
def make_mixed_image(img,mask,instance_img):
    
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
    
    #복원된 합성된 이미지 반환
    return recon_img

path=os.getcwd()+'/'

Train=Read_Data(path=path)
Train=np.array(Train)

img = np.array(Image.open(Train[0][0]))
mask = np.array(Image.open(Train[0][1]))

instance_img=make_instance(mask,img,1)

#10개 테스트
for i in range(1,11):
    img2 = np.array(Image.open(Train[i][0]))
    mask2 = np.array(Image.open(Train[i][1]))
    mixed_img=make_mixed_image(img2,mask2,instance_img)
    save_numpy_image('mixed_'+str(i)+'.png',mixed_img)
    
