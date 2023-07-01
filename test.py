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


class BoxDetect():
    def __init__(self , save_video = 'False'):
        # self.Camera_cofig()
        self.model = self.Model_cofig()
    
        self.save_video = save_video
        self.path = Path(os.path.relpath(ROOT, Path.cwd()))
        
        self.save_img_path = Path(r'C:\Users\11kkh\Desktop\realsense_custom_data')
        
        self.depth_min = 0.01 #meterp
        self.depth_max = 2.0 #meter

        # self.Aruco_detect()
        # self.SetCameraConfig()
        
        
    def Camera_cofig(self):
        start = time.time()
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 640*480 , 1024*768
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 640*360 , 640*480 , 960*540 , 1280*720 , 1920*1080
        self.profile = self.pipeline.start(config)
        
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()    
        self.depth_intrin = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics() # depth 카메라의 내부 파라미터
        self.color_intrin = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics() # color 카메라의 내부 파라미터

        self.depth_to_color_extrin =  self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( self.profile.get_stream(rs.stream.color)) # depth => color 외부파라미터
        self.color_to_depth_extrin =  self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( self.profile.get_stream(rs.stream.depth)) # color => depth 외부 파라미터
        
        
        self.hight_compensation_value = 0.04 # 4cm
        
        frames = self.pipeline.wait_for_frames()
            
        depth_frame = frames.get_depth_frame() # depth_frame is a 640x480 depth image
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.04), cv2.COLORMAP_HSV)
        # color_image
        origin_color_image = np.asanyarray(color_frame.get_data())
        end =  time.time() - start
        print("카메라 초기화 까지 걸린시간" , end)
        
        # self.pipeline.stop()
        # self.GetCameraConfig()
        return origin_color_image  , depth_colormap
    def Model_cofig(self): ####################################################################### 변수 초기화
        # weights = ROOT / 'config/best_m_5_27_50.pt'
        weights = ROOT / 'config/best_s_5_17_100.pt'
        
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
            
            depth_point[2] += 0.272 # 마커를 바닥에 붙힐때 지면과 공압 그리퍼 말단부와 떨어진 거리를 빼주어야 말단부의 Z기준점이 생성된다. 
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
        
    
    def CalculateAngle(self , xyxy , depth_frame , center_distance , depth_colormap):
        """first_pick box의 회전 

        Args:
            xyxy (int): 좌측 하단 color 좌표 (x1,y1) , 우측 하단 color 좌표(x2,y2)
            depth_frame : depth_frame
            center_distance (float): first_pick box의 중심으로 부터 떨어진 거리 
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
        
        # 전처리 pp
        # depth_rect_original = cv2.morphologyEx(depth_rect , cv2.MORPH_OPEN , (5,5) ,iterations=2)
        contours , hierarchy = cv2.findContours(depth_rect_one_channel, mode = cv2.RETR_EXTERNAL , method = cv2.CHAIN_APPROX_SIMPLE ) 
        largest_contour = max(contours, key=cv2.contourArea) # [contour 갯수 , 1 , 좌표]
        depth_rect_3channel = cv2.cvtColor(depth_rect_one_channel , cv2.COLOR_GRAY2RGB) # 1채널 => 3채널 변경
        
        
        cv2.rectangle(depth_colormap , (depth_pixel1[0] , depth_pixel1[1]) , (depth_pixel2[0] , depth_pixel2[1]) ,(0,0,255) , 1 ) # detph color 맵에 yolo모델의 bouding box시각화(red)
        cv2.drawContours(depth_rect_3channel, [largest_contour] , -1 , (255,0,0) , 1 )# depth_rect 시각화(Blue)

        ## 방법2 근사화
        rect = cv2.minAreaRect(largest_contour)
        box_float = cv2.boxPoints(rect)
        box_int = np.intp(np.round(box_float)) # box 좌표 정수화
        cv2.drawContours(depth_rect_3channel , [box_int] , -1 , (0,0,255) , 3 ) # 근사화한 사각형에 대한 시각화 
        
        # 
        first_pixel , second_pixel , third_pixel , last_pixel = box_int
        depth = center_distance/100 # center_distance는 cm단위여서 m단위로 변경 => 근사화한 box의 4개 좌표와 중심 depth를 가지는 점에서의 포인트 좌표로 변환 
        first_Point , second_Point , third_Point , last_Point = [rs.rs2_deproject_pixel_to_point(self.depth_intrin, [int(i[0]), int(i[1])], depth) for i in [first_pixel , second_pixel , third_pixel , last_pixel]]
        

        ## 회전방향 확인 => 각 꼭짓점 3차원 변환 후 회전각도 계산
        angle_point = 0
        
        ## 3D 포인터로 변경 후 각도 계산
        if first_Point[1] > second_Point[1] : # 회전이 발생
            if  ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) < ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):
                angle_point = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi+90
            elif ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) > ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):
                angle_point = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi
            else : # 길이 같은 경우 
                angle_point = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi # 예각, 둔각 다 가능하지만 예각만 회전
                
        elif first_Point[1] == second_Point[1]: ## 회전하지 않은 경우
            if  ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) < ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2): # height > width 
                angle_point = 90
            elif ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) > ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):
                angle_point = 0
            else : # height == width 
                angle_point = 0
        
        angle_point = round(angle_point,1)

        print(f"angle_point {angle_point}")
        
        ## 최종 center 좌표 검사 => 4개 꼭짓점 x,y의 평균으로 rect상에서 중심점 구하기 => 실제 depth 프레임에서 중심점 => 
        center = [int(round(i,0)) for i in [np.mean(box_float[:,0]) , np.mean(box_float[:,1])]]  # depth rect에서의 중심점
        new_center_depth =[center[0] + depth_pixel1[0]-spare_roi , center[1] + depth_pixel1[1] - spare_roi]# depth frame에서의 depth 좌표 (적재물 중심 좌표)
        
        
        cv2.circle(depth_colormap , new_center_depth , 1 , (0,0,255) , -1)
        # depth => depth_point 계산
        _ , new_center_point = self.DeProjectDepthPixeltoDepthPoint(new_center_depth[0] , new_center_depth[1] , depth_frame , distance = depth)
        
        # depth point로 기준점으로 부터 움직여야 하는 최종 거리 계산
        new_depth_point = ( round(self.x_ref - new_center_point[1]*100,1) , 
                             round(self.y_ref - new_center_point[0]*100 , 1) ,
                              round(self.z_ref - new_center_point[2]*100,1))
        # print(f"pre depth_center_pixel = {(depth_pixel1[0] + depth_pixelpp2[0])/2 , (depth_pixel1[1] + depth_pixel2[1])/2 }")
        # print(f"new depth_center_pixel = {new_center_depth }")
        # print(f"new_center_point : {new_center_point}") # 새롭게 생긴 박스의 중심점을 카메라 중심 기준 3D 좌표계
        # print(f"new_move_point_depth : {new_depth_point}") # 3D 기준점 , 새롭게 생긴 3D 박스 중심과 떨어진 거리 
        
        # depth_point => color_point 변화 
        # new_color_point = self.depth_point_to_color_point(new_center_point)
        # print(f"new_color_point : {new_color_point}")
        
        # new_color_point = ( round(abs(self.x_ref_color - round(new_color_point[1]*100,1)),1) , 
        #                      round(abs(self.y_ref_color - round(new_color_point[0]*100 , 1)),1) ,
        #                       round(abs(self.z_ref_color - round(new_color_point[2]*100,1)),1) )
        # # print(f"pre depth_center_pixel = {(depth_pixel1[0] + depth_pixel2[0])/2 , (depth_pixel1[1] + depth_pixel2[1])/2 }")
        # # print(f"new depth_center_pixel = {new_center_depth }")
        # print(f"new_move_point_color : {new_color_point}")
        
        
        
        return depth_colormap , depth_rect ,depth_rect_one_channel , angle_point , new_depth_point
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
        color_pixel = rs.rs2_projectrr_point_to_pixel(self.color_intrin, color_point)
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
  
           
    def Run(self,
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
            
            origin_color_image , depth_colormap = self.Camera_cofig()
            start = time.time()
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            # ## align 
            # frames = self.pipeline.wait_for_frames()
            
            
            # print("first_break point time " , time.time()- start)
            # # # Align the depth frame to color frame
            

            # # # Get aligned frames
            # depth_frame = frames.get_depth_frame() # depth_frame is a 640x480 depth image
            # color_frame = frames.get_color_frame()
            # print("frame 가져오는데 걸리는 시간: " ,  time.time()-start)
            # # depth image
            # depth_image = np.asanyarray(depth_frame.get_data())
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.04), cv2.COLORMAP_HSV)
            # # color_image
            # origin_color_image = np.asanyarray(color_frame.get_data())
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
                
            print("모델 추론 중 초기화 시간 : " , time.time()-start)
            infer_start = time.time()
            pred = model(im, augment=augment, visualize=visualize)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            infer_end = time.time() - infer_start
            print("모델 추론 시간 ::: " ,infer_end)
            centers = []
            distances = []
            labels = []
            x1y1x2y2 = []
            
            
            # Process predictions
            for i, det in enumerate(pred):  # per image ################### pred에 좌표값(x,y ,w ,h , confidence? , class값이 tensor로 저장되어 있고 for문을 통해 하나씩 불러옴.)
                seen += 1 # 처음에 seen은 0으로 저장되어 있음.
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
                            print(center)
                            depth_center_pixel =self.project_color_pixel_to_depth_pixel(center , depth_frame)
                            depth_center_pixel = list(map(lambda x : int(round(x,0)) , depth_center_pixel))
                            distance = depth_frame.get_distance(depth_center_pixel[0] , depth_center_pixel[1])
                            
                            c = int(cls)  # integer 
                            label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, distance,label, color=colors(c, True) ) 
                            cv2.circle(im0 , (center) , 3 , (255,255,255) , 3) # 중심좌표 시각화
                            
                            
                            cv2.circle(depth_colormap , depth_center_pixel , 3 , (255,255,255) , 3 ) #  depth 상의 중심좌표 시각화

                            
                            # if distance < 0.9 and center[0] < 529:
                            centers.append(center)
                            distances.append(round(distance,3))
                            x1y1x2y2.append([x1,y1,x2,y2])
                            # if self.names[c] ==
                            labels.append(self.names[c])
                            
                                    
                # Stream results (webcam 화면 으로 부터받은 결과를 출력)
                im0 = annotator.result()
                
            ## 우선순위 정하기
            results = {
                "idx" : list(range(len(centers))),
                "x1y1x2y2" : x1y1x2y2,
                "center" : centers,
                "center_x" : [centers[i][0] for i in range(len(centers))],
                "center_y" : [centers[i][1] for i in range(len(centers))],
                "distance" : distances,
                "label" : labels
            }
            print(results)
            final_idx = None
            
            # 1개 인식했는데 pallet인경우 => 디팔레타이징 종료
            # 1개 인식했지만 박스만 인식한 경우 => 없을 것 같지만(계속 동작)
            # 여러개 인식했지만 거기에 팔렛트 , 박스 포함된 경우 => 지금 그대로 
            
            
            ################################################################ y 중심값이 비슷한 경우에는 y가 우선이 아닌 x가 우선이 되도록
            if len(results['idx']) > 1 :
                df = pd.DataFrame(results) # result를 데이터 프레임으로 만듬
                df = df.loc[df['label']=='box'] # 'box'로 인식한것만 저장
                df= df.sort_values(by = ['center_y' , 'center_x']) # center_y , center_x로 정렬 
                # for i in range(len(df)):
                #     cv2.putText(origin_color_image , str(df['idx'][i]) , (df['center_x'][i] , df['center_y'][i]) , cv2.FONT_HERSHEY_COMPLEX , 1 , (255,255,255))
                min_distance_idx = df.iloc[np.argmin(df['distance'].values), 0]
                min_distance = df['distance'][min_distance_idx]
                min_xy_idx = df.iloc[0,0]
            
                if min_distance_idx == min_xy_idx:
                    final_idx = min_distance_idx
                    cv2.putText(origin_color_image , "first" , (df['center'][final_idx][0]-40 , df['center'][final_idx][1]+10) , cv2.FONT_ITALIC,1.4,(255,0,0),3 )
                else:
                    for i in df.index:
                        if i != min_distance_idx:
                            if df['distance'][i] - self.hight_compensation_value > min_distance: # hight_compensation_value:4cm(최소 최상단 박스보다 4cm 아래에 있는것은 제외)
                                df.drop(index = i , axis = 0 , inplace=True)
                    final_idx = df.iloc[0,0]
                    cv2.putText(origin_color_image , "first" , (df['center'][final_idx][0]-40 , df['center'][final_idx][1]+10) , cv2.FONT_ITALIC,1.4,(255,0,0),3 )
                
                first_pick_depth_pixel = self.project_color_pixel_to_depth_pixel(df['center'][final_idx] , depth_frame )
                _ , first_pick_depth_point = self.DeProjectDepthPixeltoDepthPoint(first_pick_depth_pixel[0] ,first_pick_depth_pixel[1] , depth_frame )
                
                # depth_point => color_point 변환 
                # first_pick_color_point = self.depth_point_to_color_point(first_pick_depth_point)
                # print(f"first_pick_color_point : {first_pick_color_point}")
                
                ## depth_point => color_point 변환x
                 # print(f"first_pick_depth_point : {first_pick_depth_point}")
                # first_pick2 = {
                #     'x' : round(abs(self.x_ref_color - round(first_pick_color_point[1]*100,1)),1),
                #     'y' : round(abs(self.y_ref_color - round(first_pick_color_point[0]*100 , 1)),1),
                #     'z' : round(abs(self.z_ref_color - round(first_pick_color_point[2]*100,1)),1),
                #     'center' : df['center'][final_idx],
                #     'x1y1x2y2' : df['x1y1x2y2'][final_idx],
                #     "depth_from_camera" : round(first_pick_color_point[2]*100,1),
                #     'label' : df['label'][final_idx]
                # }
                
                first_pick = {
                    'x' : round(abs(self.x_ref - round(first_pick_depth_point[1]*100,1)),1),
                    'y' : round(abs(self.y_ref - round(first_pick_depth_point[0]*100 , 1)),1),
                    'z' : round(abs(self.z_ref - round(first_pick_depth_point[2]*100,1)),1),
                    'center' : df['center'][final_idx],
                    'x1y1x2y2' : df['x1y1x2y2'][final_idx],
                    "depth_from_camera" : round(first_pick_depth_point[2]*100,1),
                    'label' : df['label'][final_idx]
                }
                print(f"frist_pick: {first_pick}")
                depth_colormap , self.depth_rect , self.depth_rect_original , angle, new_move_point = self.CalculateAngle(first_pick['x1y1x2y2'] , depth_frame , first_pick['depth_from_camera'], depth_colormap)
                print(f"new_move_point_depth : {new_move_point}")
                
            elif len(results['idx']) == 1: # 총 1개만 인식한 경우 
                if results["label"][0] == 'box': # 1개 인식했는데 box인경우 
                    cv2.putText(origin_color_image , "first" , (results['center'][0][0]-40 , results['center'][0][1]+10) , cv2.FONT_ITALIC,1.4,(255,0,0),3 )
                    first_pick_depth_pixel = self.project_color_pixel_to_depth_pixel(results['center'][0] , depth_frame )
                    _ , first_pick_depth_point = self.DeProjectDepthPixeltoDepthPoint(first_pick_depth_pixel[0] ,first_pick_depth_pixel[1] , depth_frame )
                    
                    ## depth_point => color_point 변환 
                    # first_pick_color_point = self.depth_point_to_color_point(first_pick_depth_point)
                    # print(f"first_pick_color_point : {first_pick_color_point}")
                    
                    ## depth_point => color_point 변환x
                    # print(f"first_pick_depth_point : {first_pick_depth_point}")
                    
                    first_pick = {
                        'x' : round(abs(self.x_ref - round(first_pick_depth_point[1]*100,1)),1),
                        'y' : round(abs(self.y_ref - round(first_pick_depth_point[0]*100 , 1)),1),
                        'z' : round(abs(self.z_ref - round(first_pick_depth_point[2]*100,1)),1),
                        'center' : results['center'][0],
                        'x1y1x2y2' : results['x1y1x2y2'][0],
                        "depth_from_camera" : round(first_pick_depth_point[2]*100,1),
                        'label' : results['label'][0]
                    }
                    print(f"frist_pick: {first_pick}")
                    depth_colormap , self.depth_rect , self.depth_rect_original ,  angle, new_move_point = self.CalculateAngle(first_pick['x1y1x2y2'] , depth_frame , first_pick['depth_from_camera'], depth_colormap)
                    print(f"new_move_point : {new_move_point}")
                    first_pick['angle'] = angle
                elif results['label'][0] == 'pallet': # 1개 인식했고, pallet만 남은경우 
                    first_pick = {
                        'x' : 0,
                        'y' : 0,
                        'z' : 0,
                        'center' : 0,
                        'x1y1x2y2' : 0,
                        "depth_from_camera" : 0,
                        'label' : 'pallet'
                    }
                    first_pick['angle'] = 0
                    cv2.putText(origin_color_image ,  "Palletizing End" , (320-230,240) , cv2.FONT_HERSHEY_DUPLEX,2,(0,0,0) , 4 , 3 )
                    print(f"frist_pick: {first_pick}")

            else: # 아무것도 인식하지 않은경우 
                cv2.putText(origin_color_image , "No box, pallet" , (320-250,240) , cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0) , 4, 2 )
            # print(first_pick)
            ## 회전 각도 확인
            
            end = time.time()
            print("모델 추론까지 total 시간 : ", end - start)
            try : 
                while 1:
                    if cv2.waitKey(1) == ord('s'):
                        now = datetime.datetime.now()
                        suffix = '.jpg'
                        file_name = f"{now.strftime('%Y_%m_%d_%H_%M_%S_%f')}"+suffix
                        file_name2 = f"{now.strftime('%Y_%m_%d_%H_%M_%S_%f')}_depth"+suffix
                        file_path = self.save_img_path / file_name
                        file_path2 = self.save_img_path / file_name2
                        image = cv2.resize(origin_color_image2 , (640,640))
                        cv2.imwrite(file_path , image)
                        # cv2.imwrite(file_path2 , self.depth_rect_original)
                        print(f'save_image : {file_name} , save_path : {file_path}')
                    if cv2.waitKey(1) == ord('f'):
                        if first_pick:
                            print(f"frist_pick : {first_pick}")
                        else: print("NO first_pick")
                    
                    #cv2.circle(im0 , (self.center_x , self.center_y) , 3, (0,0,255) , -1) # aruco marker 중심점 표시
                    cv2.imshow("original", origin_color_image2)
                    cv2.imshow("first" , origin_color_image)
                    cv2.imshow(str("result"), im0)
                    cv2.imshow("depth" ,depth_colormap)
                    
                    # cv2.imshow("dpeth_crop" , self.depth_rect)

                        

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
   model = BoxDetect(**opt)
   while 1:
        key = keyboard.read_key()
        if key == 'p':
            print("p 방향키 누름")
            model.Run(model=model.model)
        if key =='q':
            break
        if key == 'r':
            model.Aruco_detect_reset()
    
   
    
    