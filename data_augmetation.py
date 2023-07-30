import albumentations as A
import numpy as np
import cv2
from pathlib import Path


# with open("./2023_03_25_23_48_21_935180.txt" , 'r') as f:

#     for line in f.readlines():
#         label, *data = line.replace('\n' , '').split(' ')
        
#         center_x , center_y, width , height = list(map(lambda x : float(x) , data)) # 문자열을 float형태로 변경 
#         center_x , center_y , width , height = list(map(lambda x : int(round(x*640,0)) , [center_x , center_y , width , height])) # 정규화된 좌표를 픽셀좌표로 다시 변환
#         p1_x, p1_y = center_x - int(width/2) , center_y - int(height/2)
#         p2_x , p2_y = center_x + int(width/2) , center_y + int(height/2)

#          angle(img,(p1_x,p1_y) , (p2_x,p2_y) , (255,255,255) , 2)
# def Get_Data(textfile_path):
#     '''
#     yolo foramt으로 정규화된 txt파일을 다시 640, 640 픽셀크기의 이미지에 맞는 좌측상단 좌표(p1_x,p1_y), 우측 하단 좌표(p2_x, p2_y)정보로 반환하는 코드
#     input : txt file의 경로
#     return : 라벨링한 객체의 갯수에 맞는 행을 가지는 리스트 반환되며 각 행에는 [label 정보의 문자열 , p1_x,p1_y ,p2_x, p2_y] 5열이 저장되어 있음
#     '''
#     data_list = []
#     with open(textfile_path , 'r') as f:
#         for line in f.readlines():
#             label, *data = line.replace('\n' , '').split(' ')
            
#             center_x , center_y, width , height = list(map(lambda x : float(x) , data)) # 문자열을 float형태로 변경 
#             center_x , center_y , width , height = list(map(lambda x : int(round(x*640,0)) , [center_x , center_y , width , height])) # 정규화된 좌표를 픽셀좌표로 다시 변환
#             p1_x, p1_y = center_x - int(width/2) , center_y - int(height/2)
#             p2_x, p2_y = center_x + int(width/2) , center_y + int(height/2)
#             data_list.append((label , p1_x,p1_y,p2_x,p2_y))
#     return data_list 
# data = Get_Data(r'./2023_03_25_23_48_21_935180.txt')

def augment_yolo(image, textfile_path , iteration=1):
    '''
    image = numpy 배열의 image
    textfile_path = image에 대한 라벨링 정보가 담긴 txt 파일의 위치
    
    return : augmented_image, original_bboxes, augmented_bboxes , pixel_bboxes
    augmented_image = 변환된 이미지
    augmented_label_bboxes = 반복횟수 만큼의 행을 가지는 변환된 이미지의 yolo format에 맞게 [label center_x , center_y , width , height] 저장됨.
    original_bboxes = txtfile_path로 부터 읽어 yolo format에 맞는 정규화된 좌표값들
    pixel_bboxes = augmented_bboxes로 부터 cv2.rectangle 시각화를 위해 반복횟수 만큼의 행과 , 변환된 이미지의 라벨정보, 좌측상단 좌표(p1_x,p1_y), 우측 하단 좌표(p2_x, p2_y)정보로 반환 
    '''
    original_bboxes = [] # yolo format에 따라 정규화된 bbox좌표값을 저장하는 리스트
    labels = []
    height, width, _ = image.shape
    with open(textfile_path , 'r') as f:
        for line in f.readlines(): # txt파일의 라인별로 읽어온다. 
            label, *data = line.replace('\n' , '').split(' ') # 마지막 문자열의 \n을 제거하고 공백별로 split한 후 리스트에 저장
            
            bboxes = list(map(lambda x : float(x) , data)) # 문자열을 float형태로 변경 
            original_bboxes.append(bboxes)
            labels.append(int(label))
    bbox_params = A.BboxParams(format = 'yolo' , label_fields=[])
    
    
    transform = A.Compose([
        A.HorizontalFlip(p = 0.4), # 좌우 대칭
        # A.Rotate(limit = 30 , p=0.7 ), # 회전
        # A.Blur(p=0.1), # blur 효과
        # A.Downscale(p=0.1 , interpolation=cv2.INTER_NEAREST), # downscale(이미지 품질 낮춤)
        A.RandomBrightnessContrast(p=0.5 , brightness_limit=0.2),# Randomly change brightness and contrast
        A.MotionBlur(p=0.1), # Apply motion blur to the input image using a random-sized kernel.
        # A.CLAHE(p=1), # Contrast Limited Adaptive Histogram Equalization to the input image
        # A.Lambda : 사용자 정의 함수
        # A.RandomRotate90(p=0.3), # 90도를 랜덤하게 회전
        # A.Affine(p=0.3 , translate_px= {"x":100}), # 아핀 변환(사용자가 지정한 중심을 기준으로 회전) , scale : 1보다 작으면 줌 , 1보다 크면 확대 , translate_px [x,y] x축방향 x, y축방향 y만큼이동
        # A.RandomCrop(height=640 , width=640 , p=1),
        A.ShiftScaleRotate(shift_limit=0.2 , scale_limit = 0.3 , rotate_limit=25, p=0.7),
        A.Flip(p=0.4) # 상하좌우 회전
        
        
    ], bbox_params=bbox_params)
    
    augmented_images = []
    augmented_label_bboxess = []
    pixel_label_bboxess = []
    
    for iter in range(iteration): # iteration에 맞게 agmentation 진행
    # 이미지 변환 (image, bboxes , class_labels) 정보 저장됨
    # bounding box 좌표는 yolo형식으로 정규화된 좌표(center_X , center_y , width , height)
        augmented = transform(image=image , bboxes=original_bboxes , class_labels=labels) # class_labels : 각각의 객체에 대한 클래스 정보 (int , str 타입 다 가능)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_label = augmented['class_labels'] # 인식한 클래스 정보 
        
        # 정규화된 bounding box 좌표를 pixel 좌표로 변환 하고 , 각각의 label정보도 저장  
        pixel_label_bboxes = [[   augmented_label[idx],
                            int(round(bbox[0] * width - bbox[2]*width/2)), 
                            int(round(bbox[1] * height -bbox[3]*height/2)),
                            int(round(bbox[0] * width + bbox[2]*width/2)) ,  
                            int(round(bbox[1] * height +bbox[3]*height/2))
                            ] for idx , bbox in enumerate(augmented_bboxes)]
        augmented_label_bboxes = []
        for idx , bbox in enumerate(augmented_bboxes):
            tmp = list(bbox) # 튜플을 리스트로 변환
            tmp.insert(0,augmented_label[idx]) # 리스트의 맨 앞에 라벨정도 저장 후 augmented_label_bbxes에 추가 
            augmented_label_bboxes.append(tmp)
            
            
        augmented_images.append(augmented_image) 
        augmented_label_bboxess.append(augmented_label_bboxes)
        pixel_label_bboxess.append(pixel_label_bboxes)
    
    # dypte를 object로 해야 문자열은 label은 문자열로 저장되고 , int 타입은 int로 저장된다. => 처음에 label을 문자열에서 => 숫자로 변경해서 'object'안적어도됨
    # dtype을 object로 하지 않으면 label이 가장 앞에 있어서 숫자도 전부 문자열로 바뀌어버림
    return np.array(augmented_images), np.array(augmented_label_bboxess , dtype='object'), np.array(original_bboxes) , np.array(pixel_label_bboxess , dtype='object')

def Check_image(images , pixel_label_bboxess):
    '''
    augment된 이미지 시각화하는 함수(키보드의 아무것을 누르면 순차적으로 이미지를 확인 할 수 있다. )
    input
    images => numpy배열 형태의 4차원 배열(총 이미지 갯수 , (BGR 정보))
    pixel_bboxess =>  numpy배열 형태의 각 이미지에 맞는 라벨값과 , pixel_bboxess정보 
    
    '''
    num_image = images.shape[0]
    for idx in range(num_image):
        for pixel_label_bboxes  in pixel_label_bboxess[idx]:
            label , x_min , y_min , x_max , y_max= pixel_label_bboxes
            cv2.rectangle(images[idx] , (x_min , y_min) , (x_max , y_max) ,(0,0,255) , 2)
            cv2.putText(images[idx] , str(label) , (x_min+20 , y_min-10) ,(cv2.FONT_HERSHEY_SCRIPT_COMPLEX) , 0.5 , (0,0,0) , 2)
            cv2.imshow(f"augmented_image{idx}" , images[idx])
        cv2.waitKey()
    cv2.destroyAllWindows()
        
# def Save_textfile(save_path = r'C:\Users\11kkh\Desktop\realsense_custom_data' ,augmented_label_bboxes = np.NAN):
    
#     '''
#     yolo format에 맞는 txt 파일 저장
#     '''
#     num =augmented_label_bboxes.shape[0]
#     save_path = Path(save_path)
#     for i in range(num):
#         num_object = augmented_label_bboxes[i].shape[0]
#         name = f'tmp{i}.txt'
        
#         with open(str(save_path / name) , 'w')as f:
#             for obj in range(num_object):
#                 text = str(augmented_label_bboxes[i][obj])[1:-1].replace('\n' , '') + '\n'
#                 f.write(text)
        
#         f.close()
        
        
    
def Save_augmented_image_label(save_img_path = r'C:\Users\11kkh\Desktop\realsense_custom_data' , augmented_imags = np.nan , augmented_label_bboxes = np.NAN):
    '''
    aument된 이미지 파일저장하는 함수
    save_img_path : 이미지 저장할 디렉토리 경로
    augmented_imags : 증강된 이미지의 배열 정보 들어옴(4차원) [이미지갯수 , R, G, B]
    augmented_label_bboxes : 증강된 이미지의 라벨 정보 들어옴
    '''
    import datetime
    num = augmented_imags.shape[0]
    now = datetime.datetime.now()
    save_img_path = Path(save_img_path)
    suffix_img = '.jpg'
    suffix_txt = '.txt'
    for idx in range(num):
        appendix_name = f"{now.strftime('%Y_%m_%d_%H_%M_%S_%f')}_{str(idx)}" # 이미지, txt 파일이름을 동일하게 하기 위한 이름 설정
        img_name = appendix_name + suffix_img # 동일한 이름.jpg
        txt_name = appendix_name + suffix_txt # 동일한 이름.txt
        
        # 이미지 저장
        file_path = save_img_path / 'check_data' /img_name
        cv2.imwrite(str(file_path) , augmented_imags[idx])
        
        
        # txt 파일 저장
        file_path = save_img_path /'check_label'/ txt_name
        num_object = augmented_label_bboxes[idx].shape[0]
        
        with open(str(file_path) , 'w')as f:
            for obj in range(num_object): # 인식한 객체의 수만큼 라벨 정보 있어야함.
                text = str(augmented_label_bboxes[idx][obj])[1:-1].replace('\n' , '') + '\n' # 라벨정보 
                f.write(text)
        f.close()
        print(f'save_image : ')

            
import cv2
import os
import glob
from pathlib import Path
import argparse



# # 이미지 로드(특정 디렉토리 모든 이미지 증강)

img_list = glob.glob(r'C:\Users\11kkh\Desktop\realsense_custom_data\data\*.jpg')
txt_list = glob.glob(r'C:\Users\11kkh\Desktop\realsense_custom_data\label\*.txt')
# 반복횟수 지정
iteration = 10
for idx , path in enumerate(img_list):
    compare = Path('C:/Users/11kkh/Desktop/realsense_custom_data/label/').joinpath(str(Path(path).stem)+".txt") 
    if not compare.exists():
        continue
    image = cv2.imread(img_list[idx])
    txt_file = compare
    
    # bounding box 좌표 설정 (형식: [x_min, y_min, x_max, y_max])

    # YOLO 형식으로 이미지와 bounding box를 증강합니다.
    augmented_images, augmented_label_bboxes,original_bboxes,pixel_label_bboxes = augment_yolo(image, txt_file , iteration=iteration)
    # Save_augmented_image(augmented_imags=augmented_images)
    # Check_image(augmented_images , pixel_label_bboxes)
    Save_augmented_image_label(augmented_imags=augmented_images ,augmented_label_bboxes=augmented_label_bboxes)

# 이미지 테스트용도 확인
# image = cv2.imread(r'C:\Users\11kkh\Desktop\realsense_custom_data\data\2023_07_08_22_51_50_728364.jpg')
# txt_file = r'C:\Users\11kkh\Desktop\realsense_custom_data\label\2023_07_08_22_51_50_728364.txt'
# iteration = 10
# # bounding box 좌표 설정 (형식: [x_min, y_min, x_max, y_max])

# # YOLO 형식으로 이미지와 bounding box를 증강합니다.
# augmented_images, augmented_label_bboxes,original_bboxes,pixel_label_bboxes = augment_yolo(image, txt_file , iteration=iteration)
# # Save_augmented_image(augmented_imags=augmented_images)
# Check_image(augmented_images , pixel_label_bboxes)
# Save_augmented_image_label(augmented_imags=augmented_images ,augmented_label_bboxes=augmented_label_bboxes)

