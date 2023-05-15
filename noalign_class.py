import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import datetime

import argparse
import os
import sys
from pathlib import Path
import torch

FILE = Path(__file__).resolve() # 현재 파일의 전체 경로 (resolve() 홈디렉토리부터 현재 경로까지의 위치를 나타냄)
ROOT = FILE.parents[0]  # YOLOv5 root directory , ROOT = 현재 파일의 부모 경로 
if str(ROOT) not in sys.path: # 시스템 path에 해당 ROOT 경로가 없으면 sys.path에 추가 
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative (오른쪽 경로를 기준으로 했을 때 왼쪽 경로의 상대경로) => 현재 터미널상 디렉토리 위치와, 현재 파일의 부모경로와의 상대경로

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams  # LoadImages랑 LoadStreams는 다시한번 보기 
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
####################################################################### camera setting
import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from utils.aruco_utils import ARUCO_DICT, aruco_display

class BoxDetect():
    def __init__(self , save_video = 'False'):
        self.Camera_cofig()
        self.model = self.Model_cofig()
    
        self.save_video = save_video
        self.path = Path(os.path.relpath(ROOT, Path.cwd()))
        
        self.save_img_path = Path(r'C:\Users\11kkh\Desktop\realsense_custom_data')
        
        self.depth_min = 0.11 #meter
        self.depth_max = 1.0 #meter
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
        
        self.cm_per_pixel_ratio =  0.159514 # pixel to cm 관련 변수 #0.09433962264150944
        self.hight_compensation_value = 0.01 # 1cm
        
        # self.GetCameraConfig()
    def Model_cofig(self): ####################################################################### 변수 초기화
        weights = ROOT / 'config/best.pt'
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
            center_x , center_y = int((x2+x1)/2) , int((y2+y1)/2)
            
            depth_frame = self.pipeline.wait_for_frames().get_depth_frame()
            depth_pixel = self.project_color_pixel_to_depth_pixel((center_x,center_y) , depth_frame)
            _, depth_point = self.DeProjectDepthPixeltoDepthPoint(depth_pixel[0] , depth_pixel[1] , depth_frame )
            self.x_ref, self.y_ref , self.z_ref  = round(depth_point[1],4) , round(depth_point[0],4) , round(depth_point[2] , 4)
            print(f"기준좌표 : {self.x_ref,self.y_ref,self.z_ref}")
            
        else:
            print("No aruco_markers")
            
            
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
            line_thickness=2,  # bounding box thickness (pixels)
            ):

        if self.save_video:
            save_path = str(self.path / "result")
            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, 2.0, (640+640, 480)) #  fps, w, h = 30, im0.shape[1], im0.shape[0]

        try:
            while True:
                seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
                ## align 
                frames = self.pipeline.wait_for_frames()
                # # Align the depth frame to color frame
                

                # # Get aligned frames
                depth_frame = frames.get_depth_frame() # depth_frame is a 640x480 depth image
                color_frame = frames.get_color_frame()

                # # Validate that both frames are valid
                if not depth_frame or not color_frame:
                    continue
                        
                # depth image
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
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
                                center = int((x1+x2)/2) , int((y1+y2)/2)
                                distance = depth_frame.get_distance(center[0] , center[1])
                                
                                c = int(cls)  # integer 
                                label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, distance,label, color=colors(c, True) ) 
                                cv2.circle(im0 , (center) , 3 , (255,255,255) , 3) # 중심좌표 시각화
                                
                                depth_center_pixel = self.project_color_pixel_to_depth_pixel(center , depth_frame)
                                depth_center_pixel = list(map(lambda x : int(round(x,0)) , depth_center_pixel))
                                cv2.circle(depth_colormap , depth_center_pixel , 3 , (255,255,255) , 3 ) #  depth 상의 중심좌표 시각화

                                
                                # if distance < 0.9 and center[0] < 529:
                                centers.append(center)
                                distances.append(round(distance,3))
                                labels.append(self.names[c])
                                x1y1x2y2.append([x1,y1,x2,y2])
                                
                                        
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
                final_idx = None
                

                if len(results['idx']) != 0:
                    df = pd.DataFrame(results).sort_values(by = ['center_y' , 'center_x'])
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
                                if df['distance'][i] - self.hight_compensation_value > min_distance:
                                    df.drop(index = i , axis = 0 , inplace=True)
                        final_idx = df.iloc[0,0]
                        cv2.putText(origin_color_image , "first" , (df['center'][final_idx][0]-40 , df['center'][final_idx][1]+10) , cv2.FONT_ITALIC,1.4,(255,0,0),3 )
                    
                    first_pick_depth_pixel = self.project_color_pixel_to_depth_pixel(df['center'][final_idx] , depth_frame )
                    _ , first_pick_depth_point = self.DeProjectDepthPixeltoDepthPoint(first_pick_depth_pixel[0] ,first_pick_depth_pixel[1] , depth_frame )
                    
                    
                    
                    first_pick = {
                        'x' : round(abs(self.x_ref - first_pick_depth_point[1])*100 , 1),
                        'y' : round(abs(self.y_ref - first_pick_depth_point[0])*100 , 1),
                        'z' : round(first_pick_depth_point[2]*100,1),
                        'center' : df['center'][final_idx],
                        'x1y1x2y2' : df['x1y1x2y2'][final_idx],
                        'label' : df['label'][final_idx]
                    }
                    print(f"frist_pick: {first_pick}")
                    # 단위 cm q
                    # if first_pick['z'] > 90: 
                    #     first_pick['label'] = "pallet"
                    # print(first_pick)
                else:
                    cv2.putText(origin_color_image , "Palletizing End" , (320-230,240) , cv2.FONT_HERSHEY_DUPLEX,2,(0,0,0) , 4 , 3 )
                    
                ## 회전 각도 확인
                # depth_colormap , depth_rect = self.CalculateAngle(first_pick['x1y1x2y2'] , depth_frame , first_pick['z'], depth_colormap)
                
                    
                if self.save_video: # 동영상 저장 
                    hstack_img = np.hstack((origin_color_image , im0))
                    out.write(hstack_img) # 동영상 저장
                
                if cv2.waitKey(1) == ord('s'):
                    now = datetime.datetime.now()
                    suffix = '.jpg'
                    file_name = f"{now.strftime('%Y_%m_%d_%H_%M_%S_%f')}"+suffix
                    file_path = self.save_img_path / file_name
                    image = cv2.resize(origin_color_image2 , (640,640))
                    cv2.imwrite(file_path , image)
                    print(f'save_image : {file_name} , save_path : {file_path}')
                    
                    
                cv2.imshow("original", origin_color_image)
                cv2.imshow(str("result"), im0)
                cv2.imshow("depth" ,depth_colormap)
                # cv2.imshow("rect" , depth_rect)
                # cv2.waitKey(1)  # 1 millisecondqq
                    

                if cv2.waitKey(1) == ord('q'):
                    break 
                  
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            if self.save_video:
                out.release()
            
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
    def DeProjectDepthPixeltoDepthPoint(self, x_depth_pixel, y_depth_pixel , depth_frame) -> float:
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
        depth = depth_frame.get_distance(int(x_depth_pixel), int(y_depth_pixel)) # depth 카메라 상의 픽셀 정보를 바탕으로 depth 갚 구함
        depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [int(x_depth_pixel), int(y_depth_pixel)], depth) # depth 카메라의 픽셀과 depth 값을 통해 3D 좌표계 구함. 
        return depth, depth_point
    # def CalculateAngle(self , xyxy , depth_frame , center_distance , depth_colormap):
    #     """first_pick box의 회전 각도 계산

    #     Args:
    #         xyxy (int): 좌측 하단 color 좌표 (x1,y1) , 우측 하단 color 좌표(x2,y2)
    #         depth_frame : depth_frame
    #         center_distance (float): first_pick box의 중심으로 부터 떨어진 거리 
    #     """
        
    #     x1 ,y1 , x2,y2 = xyxy
        
    #     depth_image = np.round(np.asanyarray(depth_frame.get_data()) * self.depth_scale,4)
    #     depth_pixel1 = self.project_color_pixel_to_depth_pixel((x1,y1) , depth_frame)
    #     depth_pixel2 = self.project_color_pixel_to_depth_pixel((x2,y2) , depth_frame)
    #     depth_pixel1 = list(map(lambda x : int(round(x,0)) , depth_pixel1))
    #     depth_pixel2 = list(map(lambda x : int(round(x,0)) , depth_pixel2))
    #     depth_rect = depth_image[depth_pixel1[1]-10 : depth_pixel2[1]+10 , depth_pixel1[0]-10 : depth_pixel2[0]+10]
    #     depth_rect = np.where((depth_rect >= center_distance/100-0.01) & (depth_rect <= center_distance/100+0.01) ,255, 0)
    #     depth_rect = depth_rect.astype(np.uint8)  
    #     contours , hierarchy = cv2.findContours(depth_rect, mode = cv2.RETR_LIST , method = cv2.CHAIN_APPROX_SIMPLE ) 
    #     for c in contours:
    #         c = np.array(c).squeeze()
    #         x = c[:,0]
    #         y = c[:,1]
    #         x_min, y_min= min(x) , min(y)
    #         x_max, y_max= max(x) , max(y)
    #         x_min_indexes = np.where(x == x_min)[0]
    #         x_max_indexes = np.where(x == x_max)[0]
    #         y_min_indexes = np.where(y == y_min)[0]
    #         y_max_indexes = np.where(y == y_max)[0]
            
    #         #p1 : y가 가장 작을때의 x값 , 그때의 x값이 여러개 존재하는 경우에 대해서는 가장 작은 x값 
    #         #p2 : x가 가장 클때의 y값 , 그때의 y값이 여러개 존재하는 경우에 대해서는 가장 작은 y값
    #         #p3 : y가 가장 클때의 x값 , 그때의 x값이 여러개 존재하는 경우에 대해서는 가장 큰 x값
    #         #p4 : x가 가장 작을때 y값 , 그때의 y값이 여러개 존재하는 경우에 대해서는 가장 큰 y값
            
    #         #p1 구하기
    #         p1_x_condidate , p1_y_condidate = x[y_min_indexes] , y[y_min_indexes]
    #         p1_x , p1_x_index = min(p1_x_condidate) , np.argmin(p1_x_condidate)
    #         p1_y = p1_y_condidate[p1_x_index]
    #         cv2.circle(depth_colormap , (p1_x+depth_pixel1[0],p1_y+depth_pixel1[1]) ,2,(0,0,0) , 2 )
    #         cv2.putText(depth_colormap , "p1" , (p1_x+depth_pixel1[0] , p1_y-10+depth_pixel1[1]) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(0,0,0))
            
    #         #p2 구하기 
    #         p2_y_condidate , p2_x_condidate = y[x_max_indexes] , x[x_max_indexes]
    #         p2_y , p2_y_index = min(p2_y_condidate), np.argmin(p2_y_condidate)
    #         p2_x = p2_x_condidate[p2_y_index]
    #         cv2.circle(depth_colormap , (p2_x+depth_pixel1[0],p2_y+depth_pixel1[1]) ,2,(0,0,0) , 2 )
    #         cv2.putText(depth_colormap , "p2" , (p2_x+depth_pixel1[0] , p2_y-10+depth_pixel1[1]) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(0,0,0))
            
    #         #p3 구하기
    #         p3_x_condidate , p3_y_condidate = x[y_max_indexes] , y[y_max_indexes]
    #         p3_x , p3_x_index = max(p3_x_condidate) , np.argmax(p3_x_condidate)
    #         p3_y = p3_y_condidate[p3_x_index]
    #         cv2.circle(depth_colormap , (p3_x+depth_pixel1[0],p3_y+depth_pixel1[1]) ,2,(0,0,0) , 2 )
    #         cv2.putText(depth_colormap , "p3" , (p3_x+depth_pixel1[0] , p3_y-10+depth_pixel1[1]) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(0,0,0))
            
    #         #p4 구하기 
    #         p4_y_condidate , p4_x_condidate = y[x_min_indexes] , x[x_min_indexes]
    #         p4_y , p4_y_index = max(p4_y_condidate), np.argmax(p4_y_condidate)
    #         p4_x = p4_x_condidate[p4_y_index]
    #         cv2.circle(depth_colormap , (p4_x+depth_pixel1[0],p4_y+depth_pixel1[1]) ,2,(0,0,0) , 2 )
    #         cv2.putText(depth_colormap , "p4" , (p4_x+depth_pixel1[0] , p4_y-10+depth_pixel1[1]) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(0,0,0))
    #         return depth_colormap , depth_rect
        
        

        # return angle 
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
        sensor_dep.set_option(rs.option.digital_gain , 1)  
        sensor_dep.set_option(rs.option.laser_power , 89)  
        sensor_dep.set_option(rs.option.confidence_threshold , 2)  
        sensor_dep.set_option(rs.option.min_distance , 490)  
        sensor_dep.set_option(rs.option.post_processing_sharpening , 1)  
        sensor_dep.set_option(rs.option.pre_processing_sharpening , 2)  
        sensor_dep.set_option(rs.option.noise_filtering , 3)  
        sensor_dep.set_option(rs.option.invalidation_bypass , 1)  

parser = argparse.ArgumentParser()   
parser.add_argument('--save_video' , action='store_true' , help='save_video')
opt = parser.parse_args()
opt = vars(opt)

if __name__ == "__main__":
#    check_requirements(exclude=('tensorboard', 'thop'))
   model = BoxDetect(**opt)
   model.Run(model=model.model)
    