import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import cv2
import keyboard
import numpy as npr
import pandas as pd
import pyrealsense2 as rs
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve() # 현재 파일의 전체 경로 (resolve() 홈디렉토리부터 현재 경로까지의 위치를 나타냄)
ROOT = FILE.parents[0]  # YOLOv5 root directory , ROOT = 현재 파일의 부모 경로 
if str(ROOT) not in sys.path: # 시스템 path에 해당 ROOT 경로가 없으면 sys.path에 추가 
    sys.path.append(str(ROOT))  # add ROOT to PATHrrrr
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative (오른쪽 경로를 기준으로 했을 때 왼쪽 경로의 상대경로) => 현재 터미널상 디렉토리 위치와, 현재 파일의 부모경로와의 상대경로

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
####################################################################### camera setting
import pyrealsense2 as rs
import time

from models.common import DetectMultiBackend
from utils.aruco_utils import ARUCO_DICT, aruco_display
from utils.dataloaders import (IMG_FORMATS,  # LoadImages랑 LoadStreams는 다시한번 보기
                               VID_FORMATS, LoadImages, LoadScreenshots,
                               LoadStreams)
from utils.general import (LOGGER, Profile, check_file, check_img_size,
                           check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args,
                           scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class yolov5_demo(): # __init__ 부분은 건드리지 말고 
    def __init__(self , save_video = 'False'):
        self.Camera_cofig()
        self.model = self.Model_cofig()
    
        self.save_video = save_video
        self.path = Path(os.path.relpath(ROOT, Path.cwd()))
        
        self.save_img_path = Path(r'C:\Users\11kkh\Desktop\realsense_custom_data')
        
        self.depth_min = 0.01 #meterp
        self.depth_max = 2.0 #meter
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()    
        self.depth_intrin = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics() # depth 카메라의 내부 파라미터
        self.color_intrin = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics() # color 카메라의 내부 파라미터

        self.depth_to_color_extrin =  self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( self.profile.get_stream(rs.stream.color)) # depth => color 외부파라미터
        self.color_to_depth_extrin =  self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( self.profile.get_stream(rs.stream.depth)) # color => depth 외부 파라미터
        
        self.Aruco_detect()
        self.SetCameraConfig()
        
        
    def Camera_cofig(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 640*480 , 1024*768
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 640*360 , 640*480 , 960*540 , 1280*720 , 1920*1080
        self.profile = self.pipeline.start(config)
        
        self.hight_compensation_value = 0.04 # 4cm
        
        # self.GetCameraConfig()
    def Model_cofig(self): ####################################################################### 변수 초기화
        weights = ROOT / 'config/best_m_5_27_50.pt'
        # weights = ROOT / 'config/best_s_5_17_100.pt'
        
        data=ROOT / 'data/coco128.yaml'
        imgsz=(640, 640)  # inference size (height, width)          
        half=False  # use FP16 half-precision inference
        dnn=False  # use OpenCV DNN for ONNX inference
        
        # Load model
        device = select_device()
        model = DetectMultiBackend( weights, device=device, dnn=dnn, data=data, fp16=half) # 앞에서 정의한 weights , device , data: 어떤 데이터 쓸 것인지
        stride, self.names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
    
        # self.view_img = check_imshow(warn=True) # cv2.imshow()명령어가 잘 먹는 환경인지 확인
        return model 
    def Aruco_detect(self):
        type = "DICT_5X5_100"
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])
        arucoParams = cv2.aruco.DetectorParameters_create()
        
        while 1: 
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            corners, ids, rejected = cv2.aruco.detectMarkers(color_image, arucoDict, parameters=arucoParams)
            
            if len(corners) ==0:
                print(f"NO aruco marker!!")
                continue
            
            # corners = np.array(corners).reshape((4, 2))
            # (topLeft, topRight, bottomRight, bottomLeft) = corners 
            
            # topRight = (int(topRight[0]), int(topRight[1]))
            # bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            # bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            # topLeft = (int(topLeft[0]), int(topLeft[1]))
            
            x1 , y1 = corners[0].reshape(4,2)[0]
            x2, y2 = corners[0].reshape(4,2)[2]
            
            
            
            
            self.center_x , self.center_y = int(round((x2+x1)/2,0)) , int(round((y2+y1)/2,0))
            
            depth_frame = self.pipeline.wait_for_frames().get_depth_frame()
            if not depth_frame : continue
            depth_pixel = self.project_color_pixel_to_depth_pixel((self.center_x,self.center_y) , depth_frame)
            
            ## 기존 방법(나중에 z_point만 기구부 길이 반영)
            _, depth_point = self.DeProjectDepthPixeltoDepthPoint(depth_pixel[0] , depth_pixel[1] , depth_frame )
        
            ## 마커 말단부 기준으로
            # depth = depth_frame.get_distance(int(round(depth_pixel[0],0)), int(round(depth_pixel[1],0))) +  0.29 # 마커와 공압 그리퍼 말단부 길이 고려(29cm)
            # depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [int(depth_pixel[0]), int(depth_pixel[1])], depth) # depth 카메라의 픽셀과
            
            depth_point[2] += 0.275 # 마커를 바닥에 붙힐때 지면과 공압 그리퍼 말단부와 떨어진 거리를 빼주어야 말단부의 Z기준점이 생성된다. 
            # 만약 => 그리퍼 길이 고려한 특정한 값을 더했는데 모델이 던져준 Z값에 공압 그리퍼가 가지 못한 경우는 특정한 값을 더 줄이면서 그리퍼 말단부가 상자에 닿을 수 있도록 해야 한다.
            
            # ## depth_point => color point 변화
            # color_point = self.depth_point_to_color_point(depth_point)
            # self.x_ref_color, self.y_ref_color , self.z_ref_color = round(color_point[1]*100,1) , round(color_point[0]*100,1) , round(color_point[2]*100 , 1)
            # print(f"기준좌표_color : {self.x_ref_color,self.y_ref_color,self.z_ref_color}")
            
            ##  depth_point => color point 변화 하지 않고
            # self.x_ref, self.y_ref , self.z_ref  = round(depth_point[1]*100,1)-2.6 , round(depth_point[0]*100,1)+2 , round(depth_point[2]*100 , 1)
            self.x_ref, self.y_ref , self.z_ref  = round(depth_point[1]*100,1) , round(depth_point[0]*100,1) , round(depth_point[2]*100 , 1)
            print(f"기준좌표 : {self.x_ref,self.y_ref,self.z_ref}")
            break
    # 그리퍼 24.5 + 공압 그리퍼 4.5 = 29cm => 모터 원점복귀 위치에서 마커랑 카메라가 일직선 상에 위치하도록 카메라 지지대 위치 조정  
    # 공압 그리퍼 중심이 모터 중심보다 1.35mm 앞에 있음. (로봇 좌표계상 x축은 조금 다르고 , y축은 동일)
    def Aruco_detect_reset(self):
        
        type = "DICT_5X5_100"
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])
        arucoParams = cv2.aruco.DetectorParameters_create()
        while 1: 
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            corners, ids, rejected = cv2.aruco.detectMarkers(color_image, arucoDict, parameters=arucoParams)
            if len(corners) ==0:
                print(f"NO aruco marker!!")
                continue
            x1 , y1 = corners[0].reshape(4,2)[0]
            x2, y2 = corners[0].reshape(4,2)[2]
            center_x , center_y = int(round((x2+x1)/2,0)) , int(round((y2+y1)/2,0))
            
            depth_frame = self.pipeline.wait_for_frames().get_depth_frame()
            if not depth_frame: continue
            depth_pixel = self.project_color_pixel_to_depth_pixel((center_x,center_y) , depth_frame)
            
            ## 기존 방법
            _, depth_point = self.DeProjectDepthPixeltoDepthPoint(depth_pixel[0] , depth_pixel[1] , depth_frame )
            
            ## 마커 말단부 기준으로

            depth = depth_frame.get_distance(int(round(depth_pixel[0],0)), int(round(depth_pixel[1],0)))  # 마커와 공압 그리퍼 말단부 길이 고려(29cm)
            depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [int(depth_pixel[0]), int(depth_pixel[1])], depth) # depth 카메라의 픽셀과
            
            depth_point[2] += 0.272
            # color_point = self.depth_point_to_color_point(depth_point)
            self.x_ref, self.y_ref , self.z_ref = round(depth_point[1]*100,1) , round(depth_point[0]*100,1) , round(depth_point[2]*100 , 1)
            print(f"기준좌표 : {self.x_ref,self.y_ref,self.z_ref}")
            
            # self.x_ref_color, self.y_ref_color , self.z_ref_color = round(color_point[1]*100,1) , round(color_point[0]*100,1) , round(color_point[2]*100 , 1)
            # print(f"기준좌표_color : {self.x_ref_color,self.y_ref_color,self.z_ref_color}")
            break
        
    def FirstPick(self, results:dict , depth_frame) ->dict :
        '''
        input : 
        1. 모델에서 인식한 결과를 dictionary 형태로 저장한 것
        2. depth_frame 
        return : df에서 최종적으로 pick 해야 하는 box에 대한 정보 return 
        '''
        final_idx = None
        # 1개 인식했는데 pallet인경우 => 디팔레타이징 종료
        # 1개 인식했지만 박스만 인식한 경우 => 없을 것 같지만(계속 동작)
        # 여러개 인식했지만 거기에 팔렛트 , 박스 포함된 경우 => 지금 그대로 
        
        # 파렛트랑 박스 각각 1개씩 인식한 경우에 대해서? 오류뜨는거 같은데 
        if len(results['idx']) > 1 :
            df = pd.DataFrame(results) # result를 데이터 프레임으로 만듬
            df = df.loc[df['label']=='box'] # 'box'로 인식한것만 저장
            min_distance_idx = df.iloc[np.argmin(df['distance'].values), 0] # 최상단 box의 인덱스 번호 
            min_distance = df['distance'][min_distance_idx] # 최상단 box와 카메라간의 떨어진 거리(m)
            # print("original df",df)
            min_y = df['center_y'].min() # 가장 작은 y값 
            max_diff = 50 # 80pixel
            df = df.loc[df['distance']-self.hight_compensation_value < min_distance ]
            df['diff'] = df['center_y'].apply(lambda x: x-min_y) # 가장 작은 y값과의 차이 저장 
            df = df.loc[df['diff']<= max_diff] # drop 되고 남은 데이터 중에서 diff가 max_diff보다 작은 것만 필터링
            df = df.sort_values(by = 'center_x') 
            final_idx = df.iloc[0,0] # 남은 것들 중에서 center_x로 재 정렬 후 가장 맨 위의 데이터로 최종 pick 후보 선정
            # print("final df" , df)
            
            first_pick_depth_pixel = self.project_color_pixel_to_depth_pixel(df['center'][final_idx] , depth_frame )
            _ , first_pick_depth_point = self.DeProjectDepthPixeltoDepthPoint(first_pick_depth_pixel[0] ,first_pick_depth_pixel[1] , depth_frame )
            
            first_pick = {
                'x' : round(abs(self.x_ref - round(first_pick_depth_point[1]*100,1)),1),
                'y' : round(abs(self.y_ref - round(first_pick_depth_point[0]*100 , 1)),1),
                'z' : round(abs(self.z_ref - round(first_pick_depth_point[2]*100,1)),1),
                'center' : df['center'][final_idx],
                'x1y1x2y2' : df['x1y1x2y2'][final_idx],
                "depth_from_camera" : round(first_pick_depth_point[2]*100,1),
                'label' : df['label'][final_idx]
            }
            return first_pick
            
        elif len(results['idx']) == 1: # 총 1개만 인식한 경우 
            # first_pick = None
            if results["label"][0] == 'box': # 1개 인식했는데 box인경우 
                first_pick_depth_pixel = self.project_color_pixel_to_depth_pixel(results['center'][0] , depth_frame )
                _ , first_pick_depth_point = self.DeProjectDepthPixeltoDepthPoint(first_pick_depth_pixel[0] ,first_pick_depth_pixel[1] , depth_frame )
                first_pick = {
                    'x' : round(abs(self.x_ref - round(first_pick_depth_point[1]*100,1)),1),
                    'y' : round(abs(self.y_ref - round(first_pick_depth_point[0]*100 , 1)),1),
                    'z' : round(abs(self.z_ref - round(first_pick_depth_point[2]*100,1)),1),
                    'center' : results['center'][0],
                    'x1y1x2y2' : results['x1y1x2y2'][0],
                    "depth_from_camera" : round(first_pick_depth_point[2]*100,1),
                    'label' : results['label'][0]
                }
            elif results['label'][0] == 'pallete': # 1개 인식했고, pallet만 남은경우 ## 모델에서 이름이 "box" , "pallete로 되어 있어서 수정해야함."
                first_pick = {
                    'x' : 0.0,
                    'y' : 0.0,
                    'z' : 0.0,
                    'center' : (0,0),
                    'x1y1x2y2' : 0,
                    "depth_from_camera" : 0,
                    'label' : 'pallet'
                }   
            return first_pick        
        else: # 아무것도 인식하지 않은경우c
            first_pick = {
                   'x' : 0.0,
                    'y' : 0.0,
                    'z' : 0.0,
                    'center' : (0,0),
                    'x1y1x2y2' : 0,
                    "depth_from_camera" : 0,
                    'label' : 'NULL'
                }
            return first_pick 
    
    def PlotFirstPick(self , first_pick , color_image):
        if first_pick['label'] =='box':
            cv2.putText(color_image , "first" , (first_pick['center'][0]-40 , first_pick['center'][1]+10) , cv2.FONT_ITALIC,1.4,(255,0,0),3 )
        elif first_pick['label'] =='pallet':
            cv2.putText(color_image ,  "Palletizing End" , (320-230,240) , cv2.FONT_HERSHEY_DUPLEX,2,(0,0,0) , 4 , 3 )
        elif first_pick['label'] == 'NULL':
            cv2.putText(color_image , "No box, pallet" , (320-250,240) , cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0) , 4, 2 )
    def FindLargeContour_BoxPoint(self , image):
        '''
        input : binary_image
        output : largest_contour , box_point 
        '''
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box_point = cv2.boxPoints(rect) 
        return  largest_contour , box_point 
    # IOU 구하기
    def FindIOU(self,binary , largest_contour , box_point):
        intersection = np.zeros_like(binary)
        cv2.drawContours(intersection, [largest_contour], -1, 255, thickness=cv2.FILLED)
        largest_contour_area = np.sum(intersection)/255

        intersection[:,:]=0
        cv2.drawContours(intersection, [box_point], -1, 255, thickness=cv2.FILLED)
        box_contour_area=np.sum(intersection)/255

        cv2.drawContours(intersection, [largest_contour], -1, 255, thickness=cv2.FILLED)
        union_area = np.sum(intersection)/255
        intersection_area = largest_contour_area +box_contour_area - np.sum(intersection)/255
        IOU = round(intersection_area/union_area , 2)
        # print("합집합 넓이 : ",union_area)
        # print("largest_contour 면적: " , largest_contour_area)
        # print("box contour 면적 : " , box_contour_area)
        # print("교집합 넓이:", intersection_area )
        print("IOU" , IOU)
        
        return IOU
    def FindAngle(self, box_point):
        first_Point , second_Point , third_Point , last_Point = box_point
        angle = 0.0
        add_90angle = False
        if first_Point[1] > second_Point[1]: # 회전한 경우 
            if  ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) < ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):# height > width 
                angle = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi+90
                add_90angle = True
            elif ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) > ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):
                angle = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi
            else : # 길이 같은 경우 
                angle = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi # 회전했는데 가로, 세로길이 같은경우에는 예각, 둔각 회전 가능하지만 예각으로 회전
        elif first_Point[1] == second_Point[1]: ## 회전하지 않은 경우
            if  ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) < ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2): # height > width 
                angle = 90.0
                add_90angle = True
            elif ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) > ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):
                angle = 0.0
            else : # height == width 
                angle = 0.0
        return round(angle,1) , add_90angle
    
    def CalculateAngle(self , xyxy , depth_frame , center_distance):
        """first_pick box의 회전 각도 계산

        Args:
            xyxy (int): 좌측 하단 color 좌표 (x1,y1) , 우측 하단 color 좌표(x2,y2)
            depth_frame : depth_frame
            center_distance (float): first_pick box의 중심으로 부터 떨어진 거리 
        return : 
            return depth_rect , result
        """
        # color 프레임에서의 픽셀
        x1 ,y1 , x2,y2 = xyxy
        
        depth_image = np.round(np.asanyarray(depth_frame.get_data()) * self.depth_scale*100,1)
        # color 프레임에 대응하는 depth 픽셀 매핑
        depth_pixel1 = self.project_color_pixel_to_depth_pixel((x1,y1) , depth_frame)
        depth_pixel2 = self.project_color_pixel_to_depth_pixel((x2,y2) , depth_frame)
        depth_pixel1 = list(map(lambda x : int(round(x,0)) , depth_pixel1))
        depth_pixel2 = list(map(lambda x : int(round(x,0)) , depth_pixel2))
        
        # 관찰할 depth 영역 확인 
        spare_roi = 3 # 10pixel
        depth_rect = depth_image[depth_pixel1[1]-spare_roi : depth_pixel2[1]+spare_roi , depth_pixel1[0]-spare_roi : depth_pixel2[0]+spare_roi]
        depth_rect = np.where((depth_rect >= center_distance-1) & (depth_rect <= center_distance+1) ,255, 0) # 이진화 작업 depth 값이 중심-2 ~ 중심+2 사이면 255(흰) 아니면 0(검)
        depth_rect_one_channel = depth_rect.astype(np.uint8)  # float => uint8로변경
        
        #### 전처리 pp
        # depth_rect_original = cv2.morphologyEx(depth_rect , cv2.MORPH_OPEN , (5,5) ,iterations=2)
        largest_contour , box_point = self.FindLargeContour_BoxPoint(depth_rect_one_channel)
        # contours , hierarchy = cv2.findContours(depth_rect_one_channel, mode = cv2.RETR_EXTERNAL , method = cv2.CHAIN_APPROX_SIMPLE ) 
        # largest_contour = max(contours, key=cv2.contourArea) # [contour 갯수 , 1 , 좌표]
        
        # cv2.rectangle(depth_colormap , (depth_pixel1[0] , depth_pixel1[1]) , (depth_pixel2[0] , depth_pixel2[1]) ,(0,0,255) , 1 ) # detph color 맵에 yolo모델의 bouding box시각화(red)
        # cv2.drawContours(depth_rect_3channel, [largest_contour] , -1 , (255,0,0) , 1 )# depth_rect 시각화(Blue)

        ######  boxPoint 근사화 
        box_int = np.intp(np.round(box_point)) # box 좌표 정수화 
        first_pixel , second_pixel , third_pixel , last_pixel = box_int
        
        ## IOU구하기
        IOU = self.FindIOU(depth_rect_one_channel , largest_contour , box_int)
        
        ## 각도 구하기 및 시각화
        depth_rect_3channel = cv2.cvtColor(depth_rect_one_channel , cv2.COLOR_GRAY2RGB) # 1채널 => 3채널 변경(시각화 이미지)
        
        angle , add_90angle = self.FindAngle(box_int)
        if IOU < 0.8: 
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (5,5))
            binary2 = cv2.erode(depth_rect_one_channel , kernel , iterations=10)
            _ , box_point2 = self.FindLargeContour_BoxPoint(binary2)
            box_int2= np.intp(box_point2)
            ## 각도 구하기 (만약 IOU가 0.8 미만인데 각도가 0도가 아닌경우 는 첫번째 , 두번째 각도의 평균으로 계산)
            angle2 , add_90angle2 = self.FindAngle(box_int2)
            print("angle_flag2 : " , add_90angle2)
            cv2.drawContours(depth_rect_3channel , [box_int2] , -1 , (0,0,255) , 2  )
            if angle < 5:
                final_angle = angle2
                print(f"angle : {angle} , angle2 : {angle2}")
                print("final_angel = angle2" , final_angle)
            else:
                final_angle = (angle+angle2)/2 
                print(f"angle : {angle} , angle2 : {angle2}")
                print("final_angel = angle+angle2" , final_angle)
                    
        else :
            final_angle = angle
        
        final_angle = round(final_angle,1)
        print("final_angle : " , final_angle)
        
        # depth = center_distance/100 # center_distance는 cm단위여서 m단위로 변경 => 근사화한 box의 4개 좌표와 중심 depth를 가지는 점에서의 포인트 좌표로 변환 
        # first_Point , second_Point , third_Point , last_Point = [rs.rs2_deproject_pixel_to_point(self.depth_intrin, [int(i[0]), int(i[1])], depth) for i in [first_pixel , second_pixel , third_pixel , last_pixel]]
        

        # ## 회전방향 확인 => 각 꼭짓점 3차원 변환 후 회전각도 계산
        # angle_point = 0

        # ## 3D 포인터로 변경 후 각도 계산
        # if first_Point[1] > second_Point[1] : # 회전이 발생
            
            
        #     if  ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) < ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2): # height > width 
        #         angle_point = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi+90
        #     elif ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) > ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):
        #         angle_point = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi
        #     else : # 길이 같은 경우 
        #         angle_point = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi # 예각, 둔각 다 가능하지만 예각만 회전
                
        # elif first_Point[1] == second_Point[1]: ## 회전하지 않은 경우
        #     if  ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) < ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2): # height > width 
        #         angle_point = 90
        #     elif ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) > ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):
        #         angle_point = 0
        #     else : # height == width 
        #         angle_point = 0
        
        # angle_point = round(angle_point,1)

        # print(f"angle_point {angle_point}")
        
        
        ############################################ depth rect를 바탕으로  실제 box 중심점 찾기
       
        ## 최종 center 좌표 검사 => 4개 꼭짓점 x,y의 평균으로 rect상에서 중심점 구하기 => 실제 depth 프레임에서 중심점 => 
        # center = [int(round(i,0)) for i in [np.mean(box_float[:,0]) , np.mean(box_float[:,1])]]  # depth rect에서의 중심점
        # new_center_depth =[center[0] + depth_pixel1[0]-spare_roi , center[1] + depth_pixel1[1] - spare_roi]# depth frame에서의 depth 좌표 (적재물 중심 좌표)
        
        
        # cv2.circle(depth_colormap , new_center_depth , 1 , (0,0,255) , -1)
        # depth => depth_point 계산
        # _ , new_center_point = self.DeProjectDepthPixeltoDepthPoint(new_center_depth[0] , new_center_depth[1] , depth_frame , distance = depth)
        
        # # depth point로 기준점으로 부터 움직여야 하는 최종 거리 계산
        # new_depth_point = ( round(self.x_ref - new_center_point[1]*100,1) , 
        #                      round(self.y_ref - new_center_point[0]*100 , 1) ,
        #                       round(self.z_ref - new_center_point[2]*100,1))
       
        
        
        result = {
            'depth_pixel' : (depth_pixel1 , depth_pixel2), # depth_colormap에서 시각화를 위한 좌표 (좌상단 , 우하단 좌표)
            'pixel' : np.array([first_pixel , second_pixel , third_pixel , last_pixel]),
            # 'Points' : (first_Point, second_Point , third_Point, last_Point),
            'angle' : final_angle,
        }
        
        return depth_rect_3channel , result
    def PlotCalculateAngle(self , result , depth_rect , depth_colormap):
        depth_pixel1  , depth_pixel2 = result['depth_pixel']
        first_pixel , second_pixel , third_pixel , last_pixel = result['pixel']
        # new_center = result['new_center']
        
        cv2.drawContours(depth_rect , [result['pixel']] , -1 , (0,255,0) , 3)
        # cv2.line(depth_rect, first_pixel , second_pixel , (255,0,0) , 1) 
        # cv2.line(depth_rect, second_pixel , third_pixel , (255,0,0) , 1) 
        
        cv2.rectangle(depth_colormap , (depth_pixel1[0] , depth_pixel1[1]) , (depth_pixel2[0] , depth_pixel2[1]) ,(0,0,255) , 1 ) # detph color 맵에 yolo모델의 bouding box시각화(red)
        # cv2.circle(depth_colormap , new_center , 1 , (0,0,255) , -1)
        
        return depth_rect , depth_colormap

    def project_color_pixel_to_depth_pixel(self , color_point, depth_frame)-> float: # color pixel점을  depth pixel로 매핑 
        '''
        ## input
        color_point : color_frame상의 임의의 픽셀좌표(x,y)
        depth_frame : depth_frame 
        ## return
        color pixel => depth pixel (실수형 픽셀 좌표 반환)(x,y) 
        '''
        depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(depth_frame.get_data(), self.depth_scale,
                                    self.depth_min, self.depth_max,
                                    self.depth_intrin, self.color_intrin, self.depth_to_color_extrin, self.color_to_depth_extrin, color_point)
        
        return depth_pixel
    
    def DeProjectDepthPixeltoDepthPoint(self, x_depth_pixel, y_depth_pixel , depth_frame , distance = None) -> float:
        '''
        depth_frame에서의 pixel좌표를 바탕으로 3D 좌표계로 변환해줌. 
        ## input 
        x_depth_pixel : depth_image 상의 임의의 x 픽셀값
        y_depth_pixel : depth_image 상의 임의의 y 픽셀값
        depth_frame = depth_frame
        
        ## return
        depth : depth_image의 (x,y)에서 카메라와 떨어진 거리 (m)
        depth_point : 카메라의 주점을 기준으로 하여 떨어진 거리 (x,y,z)m => 음수가 나올 수도 있음.
        '''
        if distance:
            depth = distance
        else:
            depth = depth_frame.get_distance(int(round(x_depth_pixel,0)), int(round(y_depth_pixel,0))) # depth 카메라 상의 픽셀 정보를 바탕으로 depth 갚 구함
        depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [int(round(x_depth_pixel,0)), int(round(y_depth_pixel,0))], depth) # depth 카메라의 픽셀과 depth 값을 통해 3D 좌표계 구함. 
        return depth, depth_point
            
    
    def depth_point_to_color_point(self , depth_point) -> float : 
        '''
        input  
        1. depth_point : depth frame 상에서의 point 점
        
        return 
        1. color_pixel : depth frame에 대응하는 color frame 상에서의 point 점 
        '''
        color_point= rs.rs2_transform_point_to_point(self.depth_to_color_extrin, depth_point)
        return color_point
    def ProjectColorPointToColorPixel(self , color_point) -> float :
        '''
        input 
        1. color_point : color_frame에서의 point 점
        return
        1. color_pixel : color_frame에서 point에 대응하는 pixel 값
        '''
        color_pixel = rs.rs2_project_point_to_pixel(self.color_intrin, color_point)
        return color_pixel
    def GetCameraConfig(self):
        sensor_dep = self.profile.get_device().first_depth_sensor()
        depth_scale = sensor_dep.get_depth_scale()
        print("Depth Scale is: " , depth_scale)
        print("min_distance : ",sensor_dep.get_option(rs.option.min_distance))
        # sensor_dep.set_option(rs.option.min_distance , 0)
        print("visual_preset : ",sensor_dep.get_option(rs.option.visual_preset))
        # sensor_dep.set_option(rs.option.visual_preset , 3)
        print("depth_offset : ",sensor_dep.get_option(rs.option.depth_offset))
        print("free_fall : ",sensor_dep.get_option(rs.option.freefall_detection_enabled))
        print("sensor_mode : ",sensor_dep.get_option(rs.option.sensor_mode))
        print("host_performance : ",sensor_dep.get_option(rs.option.host_performance))
        print("max_distance : ",sensor_dep.get_option(rs.option.enable_max_usable_range))
        print("noise_estimation: ",sensor_dep.get_option(rs.option.noise_estimation))
        print("alternate_ir: ",sensor_dep.get_option(rs.option.alternate_ir))
        print("digital_gain: ",sensor_dep.get_option(rs.option.digital_gain))
        print("laser_power: ",sensor_dep.get_option(rs.option.laser_power))
        print("confidence_threhold: ",sensor_dep.get_option(rs.option.confidence_threshold))
        print("mid distance: ",sensor_dep.get_option(rs.option.min_distance))
        print("receiver_gain: ",sensor_dep.get_option(rs.option.receiver_gain))
        print("post_processing_shapening: ",sensor_dep.get_option(rs.option.post_processing_sharpening))
        print("pre_processing_sharpening: ",sensor_dep.get_option(rs.option.pre_processing_sharpening))
        print("noise_filtering: ",sensor_dep.get_option(rs.option.noise_filtering))
    def SetCameraConfig(self):
        sensor_dep = self.profile.get_device().first_depth_sensor()
        sensor_dep.set_option(rs.option.min_distance , 490)
        sensor_dep.set_option(rs.option.visual_preset , 0.0)  
        sensor_dep.set_option(rs.option.error_polling_enabled , 1)  
        sensor_dep.set_option(rs.option.enable_max_usable_range , 0.0)  
        sensor_dep.set_option(rs.option.digital_gain , 0)  
        sensor_dep.set_option(rs.option.laser_power , 89)  
        sensor_dep.set_option(rs.option.confidence_threshold , 2)  
        sensor_dep.set_option(rs.option.min_distance , 490)  
        sensor_dep.set_option(rs.option.post_processing_sharpening , 1)  
        sensor_dep.set_option(rs.option.pre_processing_sharpening , 2)  
        sensor_dep.set_option(rs.option.noise_filtering , 2)  
        sensor_dep.set_option(rs.option.invalidation_bypass , 1)  
    # def Transform_depthpoint_color_pixel(self,depth_point):
    #     color_point = self.depth_point_to_color_point(depth_point):
    #     color_pixel = self.
    #     return 
        
    #     def depth_point_to_color_point(self , depth_point) -> float : 
    #     '''
    #     input  
    #     1. depth_point : depth frame 상에서의 point 점
        
    #     return 
    #     1. color_pixel : depth frame에 대응하는 color frame 상에서의 point 점 
    #     '''
    #     color_point= rs.rs2_transform_point_to_point(self.depth_to_color_extrin, depth_point)
    #     return color_point
  
           
    def Predict(self,
            model,
            augment = False,
            visualize = False,
            conf_thres = 0.65,
            iou_thres = 0.5,
            classes = None,
            agnostic_nms = False,
            max_det=1000,
            webcam = True,
            hide_labels=False,  
            hide_conf=False,  
            line_thickness=1,  # bounding box thickness (pixels)
            ):
            
            # seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            ## align 
            
            frames = self.pipeline.wait_for_frames()
            
            # # Align the depth frame to color frame
            

            # # Get aligned frames
            depth_frame = frames.get_depth_frame() # depth_frame is a 640x480 depth image
            color_frame = frames.get_color_frame()

            # depth image
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.04), cv2.COLORMAP_HSV)
            # color_image
            origin_color_image = np.asanyarray(color_frame.get_data())
            # origin_color_image = cv2.resize(origin_color_image , (640,480))
            origin_color_image2 = origin_color_image.copy() # 사진 저장을 위한 이미지
            
            color_image = np.expand_dims(origin_color_image, axis=0)
            im0s = color_image.copy()
            im = torch.from_numpy(color_image).to(model.device)
            im = im.permute(0,3,1,2)
            
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            pred = model(im, augment=augment, visualize=visualize)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        

            centers = []
            distances = []
            labels = []
            x1y1x2y2 = []
            
            # Process predictions
            for i, det in enumerate(pred):  # per image ################### pred에 좌표값(x,y ,w ,h , confidence? , class값이 tensor로 저장되어 있고 for문을 통해 하나씩 불러옴.)
                
                if webcam:  # batch_size >= 1
                    im0 =  im0s[i].copy()
                else:
                    im0 = im0s.copy()
                annotator = Annotator(im0, line_width=line_thickness, example=str(self.names)) # utils/plot의 Annotator 클래스 
                if len(det):
                    
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round() # det에 저장된 6개의 값 중에서 4개(x1,y1,x2,y2)만 불러와서 실수값의 좌표값을 반올림 작업
                    # Write results ################################################################### 결과 이미지 만들기 
                    for (*xyxy, conf, cls) in det: ## det에 담긴 tensor를 거꾸로 루프 돌리기  xyxy : det의 (x1,y1,x2,y2) , conf :얼마나 확신하는지 ,  cls: 예측한 클래스 번호 
                        if True or save_crop or view_img:  # Add bbox to image
                            x1 , y1 , x2,y2 = list(map(int , xyxy ))
                            
                            center =  int(round((x1+x2)/2,0)) , int(round((y1+y2)/2,0))
                            # print(center)
                            depth_center_pixel =self.project_color_pixel_to_depth_pixel(center , depth_frame)
                            depth_center_pixel = list(map(lambda x : int(round(x,0)) , depth_center_pixel))
                            distance = depth_frame.get_distance(depth_center_pixel[0] , depth_center_pixel[1])
                            
                            c = int(cls)  # integer 
                            label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True) ) 
                            cv2.circle(im0 , (center) , 3 , (255,255,255) , 3) # 중심좌표 시각화
                            
                            
                            cv2.circle(depth_colormap , depth_center_pixel , 3 , (255,255,255) , 3 ) #  depth 상의 중심좌표 시각화

                            
                            # if distance < 0.9 and center[0] < 529:
                            centers.append(center)
                            distances.append(round(distance,3))
                            x1y1x2y2.append([x1,y1,x2,y2])
                            labels.append(self.names[c])
                            
                                    
                # Stream results (webcam 화면 으로 부터받은 결과를 출력)
                im0 = annotator.result()
                
            ## 인식한 결과 dict형태로 저장 
            results = {
                "idx" : list(range(len(centers))),
                "x1y1x2y2" : x1y1x2y2,
                "center" : centers,
                "center_x" : [centers[i][0] for i in range(len(centers))],
                "center_y" : [centers[i][1] for i in range(len(centers))],
                "distance" : distances,
                "label" : labels
            }
            # print(results)
            ## 우선순위 정하기 
            first_pick = self.FirstPick(results , depth_frame)
            print("first_pick _ self.firstpick return value : ",first_pick)
            if first_pick['label'] == 'box':
                self.depth_rect , result = self.CalculateAngle(first_pick['x1y1x2y2'] , depth_frame , first_pick['depth_from_camera'])
                self.depth_rect , depth_colormap = self.PlotCalculateAngle(result, self.depth_rect , depth_colormap)  
                first_pick['angle'] = result['angle']
            else :
                first_pick['angle'] = 0
            
            self.PlotFirstPick(first_pick,origin_color_image)
            

            
            try : 
                while 1:
                    ## 이미지 저장
                    if cv2.waitKey(1) == ord('s'):
                        now = datetime.datetime.now()
                        suffix = '.jpg'
                        file_name = f"{now.strftime('%Y_%m_%d_%H_%M_%S_%f')}"+suffix
                        file_name2 = f"{now.strftime('%Y_%m_%d_%H_%M_%S_%f')}_depth"+suffix
                        file_path = self.save_img_path / file_name
                        file_path2 = self.save_img_path / file_name2
                        image = cv2.resize(origin_color_image2 , (640,640))
                        # cv2.imwrite(file_path , image)
                        cv2.imwrite(file_path2 , self.depth_rect)
                        print(f'save_image : {file_name} , save_path : {file_path}')
                    if cv2.waitKey(1) == ord('f'):
                        if first_pick:
                            print(f"frist_pick : {first_pick}")
                        else: print("NO first_pick")
                    
                    #cv2.circle(im0 , (self.center_x , self.center_py) , 3, (0,0,255) , -1) # aruco marker 중심점 표시
                    cv2.imshow("original", origin_color_image2)
                    cv2.imshow("first" , origin_color_image)
                    cv2.imshow(str("result"), im0)
                    cv2.imshow("depth" ,depth_colormap)
                    if first_pick['label'] == 'box':
                        cv2.imshow("dpeth_crop" , self.depth_rect)

                        

                    if cv2.waitKey(1) == ord('c'):
                        print("창 닫음")
                        break 
            finally :
                cv2.destroyAllWindows()   
            
   

parser = argparse.ArgumentParser()   
parser.add_argument('--save_video' , action='store_true' , help='save_video')
opt = parser.parse_args()
opt = vars(opt)

if __name__ == "__main__":
#    check_requirements(exclude=('tensorboard', 'thop'))
   model = yolov5_demo(**opt)
   while 1:
        key = keyboard.read_key()
        if key == 'p':
            model.Predict(model=model.model)
        if key =='q':
            break
        if key == 'r':
            model.Aruco_detect_reset()
    
   
    
    
