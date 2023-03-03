import argparse
import os
import platform
import sys
from pathlib import Path
import pyrealsense2 as rs
import torch

FILE = Path(__file__).resolve() # 현재 파일의 전체 경로 (resolve() 홈디렉토리부터 현재 경로까지의 위치를 나타냄)
ROOT = FILE.parents[0]  # YOLOv5 root directory , ROOT = 현재 파일의 부모 경로 
if str(ROOT) not in sys.path: # 시스템 path에 해당 ROOT 경로가 없으면 sys.path에 추가 
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative (오른쪽 경로를 기준으로 했을 때 왼쪽 경로의 상대경로) => 현재 터미널상 디렉토리 위치와, 현재 파일의 부모경로와의 상대경로
print('ROOT: ' ,ROOT)

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


class BoxDetect():
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1024,768, rs.format.z16, 30) # 640*480 , 1024*768
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 640*360 , 640*480 , 960*540 , 1280*720 , 1920*1080
        self.profile = self.pipeline.start(config)
        
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        sensor_dep = self.profile.get_device().first_depth_sensor()
        sensor_dep.set_option(rs.option.min_distance , 0)
        sensor_dep.set_option(rs.option.visual_preset , 3)
        
        
    def Model(self): ####################################################################### 변수 초기화
        weights = ROOT / 'runs/best.pt'
        data=ROOT / 'data/coco128.yaml'
        webcam = True
        imgsz=(640, 640)  # inference size (height, width)          
        half=False  # use FP16 half-precision inference
        dnn=False  # use OpenCV DNN for ONNX inference
        device = select_device()
        model = DetectMultiBackend( weights, device=device, dnn=dnn, data=data, fp16=half) # 앞에서 정의한 weights , device , data: 어떤 데이터 쓸 것인지
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # # Dataloader
        bs = 1  # batch_size
        view_img = check_imshow(warn=True) # cv2.imshow()명령어가 잘 먹는 환경인지 확인
    def run(self):


####################################################################################################
def run(model = model ,
        augment = False,
        visualize = False,
        conf_thres = 0.5,
        iou_thres = 0.45,
        classes = None,
        agnostic_nms = False,
        max_det=1000,
        webcam = webcam,
        view_img = view_img,
        hide_labels=False,  
        hide_conf=False,  
        line_thickness=2  # bounding box thickness (pixels)
        ):     
    try:
       
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            ## align 
            frames = pipeline.wait_for_frames()
             # # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
                    
            # depth image
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # color_image
            origin_color_image = np.asanyarray(color_frame.get_data())
            # color_image = cv2.resize(color_image , (640,480))
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
            
            results = {} # 결과 기록 변수
            # Process predictions
            for i, det in enumerate(pred):  # per image ################### pred에 좌표값(x,y ,w ,h , confidence? , class값이 tensor로 저장되어 있고 for문을 통해 하나씩 불러옴.)
                seen += 1 # 처음에 seen은 0으로 저장되어 있음.
                if webcam:  # batch_size >= 1
                    im0 =  im0s[i].copy()
                else:
                    im0 = im0s.copy()
                annotator = Annotator(im0, line_width=line_thickness, example=str(names)) # utils/plot의 Annotator 클래스 
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round() # det에 저장된 6개의 값 중에서 4개(x1,y1,x2,y2)만 불러와서 실수값의 좌표값을 반올림 작업

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class 
                    # Write results ################################################################### 결과 이미지 만들기 
                    for *xyxy, conf, cls in reversed(det): ## det에 담긴 tensor를 거꾸로 루프 돌리기  xyxy : det의 (x1,y1,x2,y2) , conf :얼마나 확신하는지 ,  cls: 예측한 클래스 번호 
                        c = int(cls)  # integer class
                        x1 , y1 , x2,y2 = xyxy # x1 , y1: 좌측 상단 좌표 , x2,y2 우측 하단 좌표
                        print(f"x : {x1 } , y : {y1} , x2: {x2} , y2: {y2} , type : {type(x1)} ")
                        center = int((x1+x2)/2) , int((y1+y2)/2)
                        ## depth 정보 가져오기(depth 카메라 해상도 (640*480))
                        distance = aligned_depth_frame.get_distance(center[0] , center[1])
                        print(distance)
                        ##
                        
                        results['center'] = center
                        results['distance'] = distance
                
            
            # 우선순위 정하기
            print(results)
            
            
            
            
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()



def inference():
    pass
    
if __name__ == "__main__":
   check_requirements(exclude=('tensorboard', 'thop'))
   run()
