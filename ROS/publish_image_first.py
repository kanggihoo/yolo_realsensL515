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
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox , FirstPickBox

from std_msgs.msg import Header
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# weights 파일 이름 변경하고, 
# 

# 모델 관련해서 이미지 받아오고 , yolov5 모델을 통해 최종 박스 정보 return
class yolov5_demo():
    def __init__(self, weights = ROOT / 'config/best_m_5_27_50.pt',
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
            depth_point[2] += 0.272 
            self.x_ref, self.y_ref , self.z_ref  = round(depth_point[1]*100,1) , round(depth_point[0]*100,1) , round(depth_point[2]*100 , 1)
            print(f"기준좌표 : {self.x_ref,self.y_ref,self.z_ref}")
            
        else:
            self.x_ref, self.y_ref , self.z_ref  = (0,0,0)
            print("No aruco_markers")

    def project_color_pixel_to_depth_pixel(self , color_point, depth_frame)-> float: 
        depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(depth_frame.get_data(), self.depth_scale,
                                    self.depth_min, self.depth_max,
                                    self.depth_intrin, self.color_intrin, self.depth_to_color_extrin, self.color_to_depth_extrin, color_point)
        
        return depth_pixel
    def DeProjectDepthPixeltoDepthPoint(self, x_depth_pixel, y_depth_pixel , depth_frame) -> float:
        depth = depth_frame.get_distance(int(x_depth_pixel), int(y_depth_pixel)) 
        depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [int(x_depth_pixel), int(y_depth_pixel)], depth) 
        return depth, depth_point
    def CalculateAngle(self , xyxy , depth_frame , center_distance , depth_colormap):
        
        
        x1 ,y1 , x2,y2 = xyxy
        
        depth_image = np.round(np.asanyarray(depth_frame.get_data()) * self.depth_scale*100,1)
        depth_pixel1 = self.project_color_pixel_to_depth_pixel((x1,y1) , depth_frame)
        depth_pixel2 = self.project_color_pixel_to_depth_pixel((x2,y2) , depth_frame)
        depth_pixel1 = list(map(lambda x : int(round(x,0)) , depth_pixel1))
        depth_pixel2 = list(map(lambda x : int(round(x,0)) , depth_pixel2))
        
        
        spare_roi = 10 # 10pixel
        depth_rect = depth_image[depth_pixel1[1]-spare_roi : depth_pixel2[1]+spare_roi , depth_pixel1[0]-spare_roi : depth_pixel2[0]+spare_roi]
        depth_rect = np.where((depth_rect >= center_distance-1) & (depth_rect <= center_distance+1) ,255, 0)
        depth_rect = depth_rect.astype(np.uint8)  
        
        # 전처리 
        depth_rect_original = cv2.morphologyEx(depth_rect , cv2.MORPH_OPEN , (5,5) ,iterations=2)
        contours , hierarchy = cv2.findContours(depth_rect, mode = cv2.RETR_EXTERNAL , method = cv2.CHAIN_APPROX_SIMPLE ) 
        largest_contour = max(contours, key=cv2.contourArea) 
        depth_rect = cv2.cvtColor(depth_rect_original , cv2.COLOR_GRAY2RGB) # 1채널 => 3채널 변경
        
        
        cv2.rectangle(self.depth_colormap , (depth_pixel1[0] , depth_pixel1[1]) , (depth_pixel2[0] , depth_pixel2[1]) ,(0,0,255) , 1 ) # detph color 맵에 yolo모델의 bouding box시각화(red)
        cv2.drawContours(depth_rect, [largest_contour] , -1 , (255,0,0) , 2 )# depth_rect 시각화(Blue)


        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(np.round(box)) # box 좌표 정수화
        cv2.drawContours(depth_rect , [box] , -1 , (0,0,255) , 3 ) # 근사화한 사각형에 대한 시각화 
        
        # 
        first_pixel , second_pixel , third_pixel , last_pixel = box
        depth = center_distance/100 
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
            if  ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) < ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2): 
                angle_point = 90
            elif ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) > ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):
                angle_point = 0
            else : # height == width 
                angle_point = 0
        
        angle_point = int(round(angle_point,0))
        # print("Angle_point" , angle_point)

        
        
        # ## 최종 center 좌표 검사 => 4개 꼭짓점 x,y의 평균으로 rect상에서 중심점 구하기 => 실제 depth 프레임에서 중심점 => 
        # center = [int(round(i,0)) for i in [np.mean(box[:,0]) , np.mean(box[:,1])]]  # depth rect에서의 중심점
        # new_center_depth =( center[0] + depth_pixel1[0]-spare_roi , center[1] + depth_pixel1[1] - spare_roi) # depth frame에서의 depth 좌표 (적재물 중심 좌표)
        
        
        # cv2.circle(self.depth_colormap , new_center_depth , 3 , (0,0,255) , -1)
        # # depth => depth_point 계산
        # _ , new_center_point = self.DeProjectDepthPixeltoDepthPoint(new_center_depth[0] , new_center_depth[1] , depth_frame , depth)
        
        # # depth point로 기준점으로 부터 움직여야 하는 최종 거리 계산
        # new_depth_point = ( round(self.x_ref - new_center_point[1]*100,1) , 
        #                      round(self.y_ref - new_center_point[0]*100 , 1) ,
        #                       round(self.z_ref - new_center_point[2]*100,1))
        # # print(f"pre depth_center_pixel = {(depth_pixel1[0] + depth_pixel2[0])/2 , (depth_pixel1[1] + depth_pixel2[1])/2 }")
        # # print(f"new depth_center_pixel = {new_center_depth }")
        # # print(f"new_center_point : {new_center_point}") # 새롭게 생긴 박스의 중심점을 카메라 중심 기준 3D 좌표계
        # # print(f"new_move_point_depth : {new_depth_point}") # 3D 기준점 , 새롭게 생긴 3D 박스 중심과 떨어진 거리 
        
        # # depth_point => color_point 변화 
        # new_color_point = self.depth_point_to_color_point(new_center_point)
        # # print(f"new_color_point : {new_color_point}")
        
        # new_color_point = ( round(abs(self.x_ref_color - round(new_color_point[1]*100,1)),1) , 
        #                      round(abs(self.y_ref_color - round(new_color_point[0]*100 , 1)),1) ,
        #                       round(abs(self.z_ref_color - round(new_color_point[2]*100,1)),1) )
        # # # print(f"pre depth_center_pixel = {(depth_pixel1[0] + depth_pixel2[0])/2 , (depth_pixel1[1] + depth_pixel2[1])/2 }")
        # # # print(f"new depth_center_pixel = {new_center_depth }")
        # print(f"new_move_point_color : {new_color_point}")
        
        
        
        return self.depth_colormap , depth_rect ,depth_rect_original , angle_point 
    
    def get_predict(self):
        ''' 
        # return
        pred: yolo 모델을 통해 예측한 object 정보
        original_image = 3채널 color image
        im0s = 4채널 color image
        '''
        frames = self.pipeline.wait_for_frames()
        

        # # Get aligned frames
        self.depth_frame = frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = frames.get_color_frame()

                
        # depth image
        depth_image = np.asanyarray(self.depth_frame.get_data())
        self.depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # color_image
        self.origin_color_image = np.asanyarray(color_frame.get_data())
        
        color_image = np.expand_dims(self.origin_color_image, axis=0)
        self.im0s = color_image.copy()
        im = torch.from_numpy(color_image).to(self.model.device)
        im = im.permute(0,3,1,2)
        
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred = self.model(im, augment=False, visualize=False)
        self.pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        

    # callback (first_pick box 정보 반환)==========================================================================
    def get_first_pick(self):
        
        centers = []
        distances = []
        labels = []
        x1y1x2y2 = []

        # Process predictions
        for i, det in enumerate(self.pred):  # per image ################### pred에 좌표값(x,y ,w ,h , confidence , class값이 tensor로 저장되어 있고 for문을 통해 하나씩 불러옴.)
            if len(det):
                for idx, (*xyxy, conf, cls) in enumerate(det): ## det에 담긴 tensor를 거꾸로 루프 돌리기  xyxy : det의 (x1,y1,x2,y2) , conf :얼마나 확신하는지 ,  cls: 예측한 클래스 번호 
                    
                    x1 , y1 , x2,y2 = xyxy 
                    
                    center = int((x1+x2)/2) , int((y1+y2)/2)
                    depth_center_pixel = self.project_color_pixel_to_depth_pixel(center , self.depth_frame)
                    depth_center_pixel = list(map(lambda x : int(round(x,0)) , depth_center_pixel))
                    distance = self.depth_frame.get_distance(depth_center_pixel[0] , depth_center_pixel[1])
                    c = int(cls)
                    centers.append(center)
                    distances.append(round(distance*100,1))
                    x1y1x2y2.append([x1,y1,x2,y2])
                    labels.append(self.names[c])
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
        final_idx = None

        if len(results['idx']) > 1 :
            df = pd.DataFrame(results) # result를 데이터 프레임으로 만듬
            df = df.loc[df['label']=='box'] # 'box'로 인식한것만 저장
            df= df.sort_values(by = ['center_y' , 'center_x']) 
            min_distance_idx = df.iloc[np.argmin(df['distance'].values), 0]
            min_distance = df['distance'][min_distance_idx]
            min_xy_idx = df.iloc[0,0]
        
            if min_distance_idx == min_xy_idx:
                final_idx = min_distance_idx
            else:
                for i in df.index:
                    if i != min_distance_idx:
                        if df['distance'][i] - self.hight_compensation_value > min_distance: 
                            df.drop(index = i , axis = 0 , inplace=True)
                final_idx = df.iloc[0,0]
            
            first_pick_depth_pixel = self.project_color_pixel_to_depth_pixel(df['center'][final_idx] , self.depth_frame )
            _ , first_pick_depth_point = self.DeProjectDepthPixeltoDepthPoint(first_pick_depth_pixel[0] ,first_pick_depth_pixel[1] , self.depth_frame )
            
            first_pick = {
                'x' : round(abs(self.x_ref - round(first_pick_depth_point[1]*100,1)),1),
                'y' : round(abs(self.y_ref - round(first_pick_depth_point[0]*100 , 1)),1),
                'z' : round(abs(self.z_ref - round(first_pick_depth_point[2]*100,1)),1),
                'center' : df['center'][final_idx],
                'x1y1x2y2' : df['x1y1x2y2'][final_idx],
                "depth_from_camera" : round(first_pick_depth_point[2]*100,1),
                'label' : df['label'][final_idx]
            }
            

            self.depth_colormap , self.depth_rect , self.depth_rect_original , angle = self.CalculateAngle(first_pick['x1y1x2y2'] , self.depth_frame , first_pick['depth_from_camera'], self.depth_colormap)
            first_pick['angle'] = angle
            
        elif len(results['idx']) == 1: # 총 1개만 인식한 경우 
            if results["label"][0] == 'box': # 1개 인식했는데 box인경우 
                first_pick_depth_pixel = self.project_color_pixel_to_depth_pixel(results['center'][0] , self.depth_frame )
                _ , first_pick_depth_point = self.DeProjectDepthPixeltoDepthPoint(first_pick_depth_pixel[0] ,first_pick_depth_pixel[1] , self.depth_frame )

                
                first_pick = {
                    'x' : round(abs(self.x_ref - round(first_pick_depth_point[1]*100,1)),1),
                    'y' : round(abs(self.y_ref - round(first_pick_depth_point[0]*100 , 1)),1),
                    'z' : round(abs(self.z_ref - round(first_pick_depth_point[2]*100,1)),1),
                    'center' : results['center'][0],
                    'x1y1x2y2' : results['x1y1x2y2'][0],
                    "depth_from_camera" : round(first_pick_depth_point[2]*100,1),
                    'label' : results['label'][0]
                }
                self.depth_colormap , self.depth_rect , self.depth_rect_original ,  angle= self.CalculateAngle(first_pick['x1y1x2y2'] , self.depth_frame , first_pick['depth_from_camera'], self.depth_colormap)
                first_pick['angle'] = angle
            elif results['label'][0] == 'pallete': # 1개 인식했고, pallete만 남은경우 
                first_pick = {
                    'x' : 0.0,
                    'y' : 0.0,
                    'z' : 0.0,
                    'center' : (0,0),
                    'x1y1x2y2' : 0,
                    "depth_from_camera" : 0,
                    "angle" : 0,
                    'label' : 'pallete'
                }

        else: # 아무것도 인식하지 않은경우 
            first_pick = {
                    'x' : 0.0,
                    'y' : 0.0,
                    'z' : 0.0,
                    'center' : ((0,0)),
                    'x1y1x2y2' : 0,
                    "depth_from_camera" : 0,
                    "angle" : 0,
                    'label' : 'no_object'
                }
           

        # print(first_pick) 
        return first_pick , results
    
    # 최종적으로 pick 해야 하는 box 결과 영상 , yolo 모델을 통해 인식한 모든 박스 영상 반환
    def get_image(self , first_pick, results):
        if len(self.pred) != 0:
            # Process predictions
            for i, det in enumerate(self.pred):  # per image ################### pred에 좌표값(x,y ,w ,h , confidence? , class값이 tensor로 저장되어 있고 for문을 통해 하나씩 불러옴.)
                # seen += 1 # 처음에 seen은 0으로 저장되어 있음.
                im0 =  self.im0s[i].copy()
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names)) # utils/plot의 Annotator 클래스 
                if len(det):
                    for idx, (*xyxy, conf, cls) in enumerate(det): ## det에 담긴 tensor를 거꾸로 루프 돌리기  xyxy : det의 (x1,y1,x2,y2) , conf :얼마나 확신하는지 ,  cls: 예측한 클래스 번호 
                        
                        x1 , y1 , x2,y2 = xyxy 
                        center =  int((x1+x2)/2) , int((y1+y2)/2)
                        c = int(cls)  # integer 
                        # label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                        label = f'{self.names[c]} {conf:.2f}'
                        annotator.box_label(xyxy,label, color=colors(c, True) ) 
                        # print(f"x : {x1 } , y : {y1} , x2: {x2} , y2: {y2} , type : {type(x1)} ")
                        
                        cv2.circle(im0 , (center) , 3 , (255,255,255) , 3) # 중심좌표 시각화                        
                                    
                # Stream results (webcam 화면 으로 부터받은 결과를 출력)
                im0 = annotator.result()
        if first_pick['label']=='box':
            center_x , center_y = first_pick['center'][0] , first_pick['center'][1]
            cv2.putText(self.origin_color_image , "first" , (center_x-40 ,center_y+10) , cv2.FONT_ITALIC,1.4,(255,0,0),3 )
        
        elif first_pick['label'] =='pallete':
            cv2.putText(self.origin_color_image ,  "Palletizing End" , (320-230,240) , cv2.FONT_HERSHEY_DUPLEX,2,(0,0,0) , 4 , 3 )
        elif first_pick['label'] =='no_object':
            cv2.putText(self.origin_color_image , "No box, pallete" , (320-250,240) , cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0) , 4, 2 )
        
        return np.hstack((self.origin_color_image , im0,self.depth_colormap))
                    


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


        # timer 선언
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.timer2 = self.create_timer(1, self.timer_callback2)


        self.yolov5 = yolov5_demo()
        self.br = CvBridge()


    # timer callback 함수(0.1초 주기로 카메라 영상 전송) 
    def timer_callback(self):
        self.yolov5.get_predict() # 영상에서의 예측결과 저장 
        self.first_pick , self.results = self.yolov5.get_first_pick()
        img = self.yolov5.get_image(self.first_pick , self.results)
        self.publisher.publish(self.br.cv2_to_imgmsg(img))
    
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
            msg.angle = self.first_pick['angle']
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


