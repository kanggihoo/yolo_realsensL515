import os
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
import pandas as pd
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 640*480 , 1024*768
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 640*360 , 640*480 , 960*540 , 1280*720 , 1920*1080
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
###################################################################### camera config setting
sensor_dep = profile.get_device().first_depth_sensor()
print("min_distance : ",sensor_dep.get_option(rs.option.min_distance))
sensor_dep.set_option(rs.option.min_distance , 0)
print("set_min_distance : ",sensor_dep.get_option(rs.option.min_distance))

print("visual_preset : ",sensor_dep.get_option(rs.option.visual_preset))
sensor_dep.set_option(rs.option.visual_preset , 3)
print("set_min_visual_preset : ",sensor_dep.get_option(rs.option.visual_preset))

####################################################################### 변수 초기화
weights = ROOT / 'runs/best.pt'
data=ROOT / 'data/coco128.yaml'
webcam = True
imgsz=(640, 640)  # inference size (height, width)          
half=False  # use FP16 half-precision inference
dnn=False  # use OpenCV DNN for ONNX inference


####################################################################### 
# (429,267) , (429,161) => 10cm
cm_per_pixel_ratio = 10/(267-161) # pixel to cm 관련 변수 #0.09433962264150944
print(cm_per_pixel_ratio)
FOV = (640*cm_per_pixel_ratio , 480*cm_per_pixel_ratio) # (카메라상 0,0 기준 시야 범위)
x_offset , y_offset = 15 , 15 # 카메라 중심으로 부터 프레임의 원점사이의 x,y 거리 (카메라상에서의 x,y 와 프레임의 x,y는 반대) 단위: cm
hight_compensation_value = 0.01
####################################################################### 
# Load model (모델 가져오기)
device = select_device()
model = DetectMultiBackend( weights, device=device, dnn=dnn, data=data, fp16=half) # 앞에서 정의한 weights , device , data: 어떤 데이터 쓸 것인지
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

 # # Dataloader
bs = 1  # batch_size
view_img = check_imshow(warn=True) # cv2.imshow()명령어가 잘 먹는 환경인지 확인

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

    while True:
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        ## align 
        frames = pipeline.wait_for_frames()
            # # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
                

        # color_image
        origin_color_image = np.asanyarray(color_frame.get_data())
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

            
                # Write results ################################################################### 결과 이미지 만들기 
                for *xyxy, conf, cls in det: #xyxy : det의 (x1,y1,x2,y2) , conf :얼마나 확신하는지 ,  cls: 예측한 클래스 번호 
                    c = int(cls)
                    x1 , y1 , x2,y2 = xyxy # x1 , y1: 좌측 상단 좌표 , x2,y2 우측 하단 좌표
                    center = int((x1+x2)/2) , int((y1+y2)/2)
                    ## depth 정보 가져오기(depth 카메라 해상도 (640*480))
                    distance = aligned_depth_frame.get_distance(center[0] , center[1])
                    centers.append(center)
                    distances.append(distance)
                    labels.append(names[c])
            
        
        # 우선순위 정하기
        results = {
            "idx" : list(range(len(centers))),
            "center" : centers,
            "center_x" : [centers[i][0] for i in range(len(centers))],
            "center_y" : [centers[i][1] for i in range(len(centers))],
            "distance" : distances,
            "label" : labels
        }
        final_idx = None
        
        if len(results) != 0:
            df = pd.DataFrame(results).sort_values(by = ['center_y' , 'center_x'])
            if len(df['distance'].values) == 0:
                continue
            min_distance_idx = df.iloc[np.argmin(df['distance'].values), 0]
            min_distance = df['distance'][min_distance_idx]
            min_xy_idx = df.iloc[0,0]
        
            if min_distance_idx == min_xy_idx:
                final_idx = min_distance_idx
            else:
                for i in df.index:
                    if i != min_distance_idx:
                        if df['distance'][i] - hight_compensation_value > min_distance:
                            df.drop(index = i , axis = 0 , inplace=True)
                final_idx = df.iloc[0,0]
            first_pick = {
                'x' : x_offset+(df['center_y'][final_idx]-239)*cm_per_pixel_ratio,
                'y' : y_offset+(319-df['center_x'][final_idx])*cm_per_pixel_ratio,
                'z' : df['distance'][final_idx]*100,
                'center' : df['center'][final_idx],
                'label' : df['label'][final_idx]
            }
            if first_pick['z'] > 90: 
                first_pick['label'] = "pallet"
            return first_pick    # 단위 cm
        
        else:
            continue
    
if __name__ == "__main__":
   check_requirements(exclude=('tensorboard', 'thop'))
   for i in range(5):   
       
       print(run())
   pipeline.stop()

        


