import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import math

import pyrealsense2 as rs
# 경로 설정
FILE = Path(__file__).resolve() # 현재 파일의 전체 경로 (resolve() 홈디렉토리부터 현재 경로까지의 위치를 나타냄)
ROOT = FILE.parents[0]  # YOLOv5 root directory , ROOT = 현재 파일의 부모 경로 

if str(ROOT) not in sys.path: # 시스템 path에 해당 ROOT 경로가 없으면 sys.path에 추가 
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative (오른쪽 경로를 기준으로 했을 때 왼쪽 경로의 상대경로) => 현재 터미널상 디렉토리 위치와, 현재 파일의 부모경로와의 상대경로

# yolo 모델 관련 라이브러리
import torch
import torch.backends.cudnn as cudnn # ?? 

from yolov5_ros.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots,LoadStreams 
from yolov5_ros.models.common import DetectMultiBackend
from yolov5_ros.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5_ros.utils.plots import Annotator, colors
from yolov5_ros.utils.torch_utils import select_device, smart_inference_mode
from yolov5_ros.utils.aruco_utils import ARUCO_DICT , aruco_display


# Ross 관련 라이브러리 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox , FirstPickBox # 여기 부분 변경하고 

from std_msgs.msg import Header
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup



# 모델 관련해서 이미지 받아오고 , yolov5 모델을 통해 최종 박스 정보 return
class yolov5_demo():
    def __init__(self, weights = ROOT / 'config/best_m_5_27_50.pt', # 가중치 파일 이름 변경하고
                        data = ROOT / 'data/coco128.yaml',
                        imagez_height = 640,
                        imagez_width = 640,
                        conf_thres = 0.65,
                        iou_thres = 0.5,
                        max_det = 1000,
                        device = 'cpu',
                        view_img = True,
                        classes = None,
                        agnostic_nms = False,
                        line_thickness = 2,
                        half = False,
                        dnn = False
                        ):
        self.weights = weights
        self.data = data
        self.imagez_height = imagez_height
        self.imagez_width = imagez_width
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.line_thickness = line_thickness
        self.half = half
        self.dnn = dnn

        #####################################3
        self.Camera_cofig()
        self.load_model()

        self.depth_min = 0.01 #meter
        self.depth_max = 2.0 #meter
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()    
        self.depth_intrin = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics() # depth 카메라의 내부 파라미터
        self.color_intrin = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics() # color 카메라의 내부 파라미터

        self.depth_to_color_extrin =  self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( self.profile.get_stream(rs.stream.color)) # depth => color 외부파라미터
        self.color_to_depth_extrin =  self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( self.profile.get_stream(rs.stream.depth)) # color => depth 외부 파라미터

        self.SetCameraConfig()
        self.Aruco_detect()

    def Camera_cofig(self)->None:
        #---------------------------------------------------------카메라 설정
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30) # 640*480 , 1024*768
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 640*360 , 640*480 , 960*540 , 1280*720 , 1920*1080
        
        self.profile = self.pipeline.start(config)
        

        # depth 보정치 
        self.hight_compensation_value = 0.04 # 1cm
        # -------------------------------------------------------------------

    def load_model(self): ######################### 모델 변수 초기화
        imgsz = (self.imagez_height, self.imagez_width)

        # load_model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(weights=self.weights , device=self.device, dnn=self.dnn, data=self.data ,fp16 = self.half) # 앞에서 정의한 weights , device , data: 어떤 데이터 쓸 것인지

        stride, self.names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        self.half &= (pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()


        source = 0
        # Dataloader
        webcam = True
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True
        bs = 1
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0], 0

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

    def Aruco_detect(self):
        frames = self.pipeline.wait_for_frames()
        # # Align the depth frame to color frame
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        type = "DICT_5X5_100"
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[type])
        arucoParams = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(color_image, arucoDict, parameters=arucoParams) 
        if len(corners):
            
            x1 , y1 = corners[0].reshape(4,2)[0]
            x2, y2 = corners[0].reshape(4,2)[2]
            center_x , center_y = int(round((x2+x1)/2,0)) , int(round((y2+y1)/2,0))
            
            depth_frame = self.pipeline.wait_for_frames().get_depth_frame()
            depth_pixel = self.project_color_pixel_to_depth_pixel((center_x,center_y) , depth_frame)
            _, depth_point = self.DeProjectDepthPixeltoDepthPoint(depth_pixel[0] , depth_pixel[1] , depth_frame )
            depth_point[2] += 0.275
            self.x_ref, self.y_ref , self.z_ref  = round(depth_point[1]*100,1) , round(depth_point[0]*100,1) , round(depth_point[2]*100 , 1)
            print(f"기준좌표 : {self.x_ref,self.y_ref,self.z_ref}")
            
        else:
            self.x_ref, self.y_ref , self.z_ref  = (0,0,0)
            print("No aruco_markers")

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
            # print(f"frist_pick: {first_pick}")
            # depth_rect , result = self.CalculateAngle(first_pick['x1y1x2y2'] , depth_frame , first_pick['depth_from_camera'])
            # print(f"new_move_point_depth : {result['new_depth_point']}")
            # first_pick['angle'] = result['angle']
            # return first_pick , depth_rect , result 
            
        elif len(results['idx']) == 1: # 총 1개만 인식한 경우 
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
                
            elif results['label'][0] == 'pallete': # 1개 인식했고, pallet만 남은경우 
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
        else: # 아무것도 인식하지 않은경우
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
        # print("IOU" , IOU)
        
        return IOU    
    def project_color_pixel_to_depth_pixel(self , color_point, depth_frame)-> float: 
        depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(depth_frame.get_data(), self.depth_scale,
                                    self.depth_min, self.depth_max,
                                    self.depth_intrin, self.color_intrin, self.depth_to_color_extrin, self.color_to_depth_extrin, color_point)
        
        return depth_pixel
    def FindLargeContour_BoxPoint(self , image):
        '''
        input : binary_image
        output : largest_contour , box_point 
        '''
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box_point = cv2.boxPoints(rect) 
        else : 
            largest_contour = 0
            box_point = 0
        return  largest_contour , box_point 
    # IOU 구하기
    def FindIOU(self,binary , largest_contour , box_point):
        if isinstance(largest_contour, np.ndarray):
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
            
        else : 
            IOU = 0
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
    def DeProjectDepthPixeltoDepthPoint(self, x_depth_pixel, y_depth_pixel , depth_frame) -> float:
        depth = depth_frame.get_distance(int(x_depth_pixel), int(y_depth_pixel)) 
        depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [int(x_depth_pixel), int(y_depth_pixel)], depth) 
        return depth, depth_point

    def CalculateAngle(self , xyxy , depth_frame , center_distance):
        '''
        first_pick box의 회전 각도 계산

        Args:
            xyxy (int): 좌측 하단 color 좌표 (x1,y1) , 우측 하단 color 좌표(x2,y2)
            depth_frame : depth_frame
            center_distance (float): first_pick box의 중심으로 부터 떨어진 거리 
        return : 
            return depth_rect , result
        '''
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
            binary2 = cv2.erode(depth_rect_one_channel , kernel , iterations=8)
            print("binary2 " , binary2.shape)
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
        
        final_angle= round(final_angle,1)
        print("final_angle : " , final_angle)
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

    
    def get_predict(self):
        ''' 
        # return
        pred: yolo 모델을 통해 예측한 object 정보
        original_image = 3채널 color image
        im0s = 4채널 color image
        '''
        frames = self.pipeline.wait_for_frames()
        

        # # Get aligned frames
        depth_frame = frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = frames.get_color_frame()

                
        # depth image
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # color_image
        origin_color_image = np.asanyarray(color_frame.get_data())
        
        color_image = np.expand_dims(origin_color_image, axis=0)
        im0s = color_image.copy()
        im = torch.from_numpy(color_image).to(self.model.device)
        im = im.permute(0,3,1,2)
        
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        centers = []
        distances = []
        labels = []
        x1y1x2y2 = []
        
        # Process predictions
        for i, det in enumerate(pred):  # per image ################### pred에 좌표값(x,y ,w ,h , confidence? , class값이 tensor로 저장되어 있고 for문을 통해 하나씩 불러옴.)
           
            im0 =  im0s[i].copy()
           
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names)) # utils/plot의 Annotator 클래스 
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
                        # label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                        label = f'{self.names[c]} {conf:.2f}'
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

            results = {
                "idx" : list(range(len(centers))),
                "x1y1x2y2" : x1y1x2y2,
                "center" : centers,
                "center_x" : [centers[i][0] for i in range(len(centers))],
                "center_y" : [centers[i][1] for i in range(len(centers))],
                "distance" : distances,
                "label" : labels
            }

            self.depth_rect = None
            first_pick = self.FirstPick(results , depth_frame)
            if first_pick['label'] == 'box':
                self.depth_rect , result = self.CalculateAngle(first_pick['x1y1x2y2'] , depth_frame , first_pick['depth_from_camera'])
                self.depth_rect , depth_colormap = self.PlotCalculateAngle(result, self.depth_rect , depth_colormap)  
                first_pick['angle'] = result['angle']
            else :
                first_pick['angle'] = 0.0
            
            self.PlotFirstPick(first_pick,origin_color_image)
            
            return first_pick , np.hstack((origin_color_image , im0,depth_colormap)) , np.zeros((10,10))
            

    

    # # callback (first_pick box 정보 반환)==========================================================================
    # def get_first_pick(self):
        
    #     return first_pick , results
    
    # 최종적으로 pick 해야 하는 box 결과 영상 , yolo 모델을 통해 인식한 모든 박스 영상 반환
    # def get_image(self , first_pick, results):
    #     if len(self.pred) != 0:
    #         # Process predictions
    #         for i, det in enumerate(self.pred):  # per image ################### pred에 좌표값(x,y ,w ,h , confidence? , class값이 tensor로 저장되어 있고 for문을 통해 하나씩 불러옴.)
    #             # seen += 1 # 처음에 seen은 0으로 저장되어 있음.
    #             im0 =  self.im0s[i].copy()
    #             annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names)) # utils/plot의 Annotator 클래스 
    #             if len(det):
    #                 for idx, (*xyxy, conf, cls) in enumerate(det): ## det에 담긴 tensor를 거꾸로 루프 돌리기  xyxy : det의 (x1,y1,x2,y2) , conf :얼마나 확신하는지 ,  cls: 예측한 클래스 번호 
                        
    #                     x1 , y1 , x2,y2 = xyxy 
    #                     center =  int((x1+x2)/2) , int((y1+y2)/2)
    #                     c = int(cls)  # integer 
    #                     # label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
    #                     label = f'{self.names[c]} {conf:.2f}'
    #                     annotator.box_label(xyxy,label, color=colors(c, True) ) 
    #                     # print(f"x : {x1 } , y : {y1} , x2: {x2} , y2: {y2} , type : {type(x1)} ")
                        
    #                     cv2.circle(im0 , (center) , 3 , (255,255,255) , 3) # 중심좌표 시각화                        
                                    
    #             # Stream results (webcam 화면 으로 부터받은 결과를 출력)
    #             im0 = annotator.result()
    #     if first_pick['label']=='box':
    #         center_x , center_y = first_pick['center'][0] , first_pick['center'][1]
    #         cv2.putText(self.origin_color_image , "first" , (center_x-40 ,center_y+10) , cv2.FONT_ITALIC,1.4,(255,0,0),3 )
        
    #     elif first_pick['label'] =='pallete':
    #         cv2.putText(self.origin_color_image ,  "Palletizing End" , (320-230,240) , cv2.FONT_HERSHEY_DUPLEX,2,(0,0,0) , 4 , 3 )
    #     elif first_pick['label'] =='no_object':
    #         cv2.putText(self.origin_color_image , "No box, pallete" , (320-250,240) , cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0) , 4, 2 )
        
    #     return np.hstack((self.origin_color_image , im0,self.depth_colormap))
                    


class ImagePublisher(Node):
    def __init__(self):
        super().__init__(node_name = 'image_publish')
        self.callback_group = ReentrantCallbackGroup()

        # image publisher 
        self.publisher = self.create_publisher(
            Image,
            'color_frame',
            10,
            callback_group=self.callback_group)
        
        # # 
        self.publisher2 = self.create_publisher(
            FirstPickBox,
            'first_pick',
            10 ,
            callback_group= self.callback_group)

        self.publisher3 = self.create_publisher(
            Image,
            'depth_frame',
            10,
            callback_group=self.callback_group)


        # timer 선언
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.timer2 = self.create_timer(timer_period, self.timer_callback2)
        self.timer3 = self.create_timer(timer_period, self.timer_callback3)


        self.yolov5 = yolov5_demo()
        self.br = CvBridge()


    # timer callback 함수(0.1초 주기로 카메라 영상 전송) 
    def timer_callback(self):
        self.first_pick , color_image, self.depth_image = self.yolov5.get_predict() # 영상에서의 예측결과 저장 
        self.publisher.publish(self.br.cv2_to_imgmsg(color_image))

    def timer_callback3(self):
        self.publisher3.publish(self.br.cv2_to_imgmsg(self.depth_image))

    # timer callback2 함수(1초마다 우선적으로 pick 해야 하는 박스 정보 전송)
    def timer_callback2(self):
        msg = FirstPickBox()
        '''
        float32 x
        float32 y
        float32 z
        int32[] center
        string class_id
        '''
        if len(self.first_pick):
            msg.x = self.first_pick['x']
            msg.y = self.first_pick['y']
            msg.z = self.first_pick['z']
            msg.center = (self.first_pick['center'][0] , self.first_pick['center'][1])
            msg.angle = round(self.first_pick['angle'],1)
            msg.class_id = self.first_pick['label']
            
        self.get_logger().info("send data : {}".format(msg))
        self.publisher2.publish(msg)



def ros_main():

    rclpy.init()
    publish_node = ImagePublisher()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(publish_node)

    try : 
        rclpy.spin(publish_node)
    except KeyboardInterrupt:
        publish_node.get_logger().info("key board Interrupt")
    finally:
        executor.shutdown()
        publish_node.destroy_node()
        rclpy.shutdown()


if __name__ =='__main__':
    ros_main()


