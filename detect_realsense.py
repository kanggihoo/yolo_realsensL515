import argparse
import os
import platform
import sys
from pathlib import Path
import pyrealsense2 as rs
import torch
import time

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
####################################################################### custom
import pyrealsense2 as rs
import numpy as np
import cv2
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 640*480 , 1024*768
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 640*360 , 640*480 , 960*540 , 1280*720 , 1920*1080
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
####################################################################### camera setting
sensor_dep = profile.get_device().first_depth_sensor()
print("min_distance : ",sensor_dep.get_option(rs.option.min_distance))
sensor_dep.set_option(rs.option.min_distance , 0)
print("set_min_distance : ",sensor_dep.get_option(rs.option.min_distance))

print("visual_preset : ",sensor_dep.get_option(rs.option.visual_preset))
sensor_dep.set_option(rs.option.visual_preset , 3)
print("set_min_visual_preset : ",sensor_dep.get_option(rs.option.visual_preset))


####################################################################### 
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL // 모델 가중치 위치
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam) // 데이터 파일위치
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path // 
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results                                ######## 요거를 True로 하면?
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        # nosave=False,  # do not save images/videos
        save=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    webcam = source.isnumeric() # souce에 들어온 값이 숫자인지 확인 후 숫자이면 webcam = True
    save_img = not save and not source.endswith('.txt')  # save inference images(save : True인경우 save_img : False)
    
    # Directories (결과를 저장할 디렉토리 만드는 과정)
    if save_img: # 결과를 저장해야 하는 경우 디렉토리 생성
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run # save_dir = 'Yolo\yolov5\runs\detect\exp14'
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # save_txt 가 true 인경우에 save_dir/labels 디렉토리 생성 아닌경우 svae_dir 디렉토리 생성
         ## 동영상 저장
        # Save results (image with detections) # 결과를 run / exp "" 파일에 결과를 저장한다(bounding 박스를 그려주고 , label이름도 그려진 이미지 저장)
        save_path = str(save_dir / "box")  # im.jpg
        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 2.0, (640+640+640, 480)) #  fps, w, h = 30, im0.shape[1], im0.shape[0]

    # Load model (모델 가져오기)
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) # 앞에서 정의한 weights , device , data: 어떤 데이터 쓸 것인지
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:          ############################## webcam인 경우 
        view_img = check_imshow(warn=True) # cv2.imshow()명령어가 잘 먹는 환경인지 확인
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # stream 할때 path : ['0'] , im.shape = (1,3,480,640) , im0s.shape = (1,480,640,3) , vid_cap : None , s : None(type: str)
    
    
    try:
        while True:
            start = time.time()
            
            ## align 
            frames = pipeline.wait_for_frames()
             # # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
                    
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
                    print( 'det[: , :4]',det[: , :4])

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class 
                        print('det[:,5]' , det[:,5] , c ) # class 값 출력

                    # Write results ################################################################### 결과 이미지 만들기 
                    for *xyxy, conf, cls in reversed(det): ## det에 담긴 tensor를 거꾸로 루프 돌리기  xyxy : det의 (x1,y1,x2,y2) , conf :얼마나 확신하는지 ,  cls: 예측한 클래스 번호 
                        if True or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True)) 
                            x1 , y1 , x2,y2 = xyxy 
                            print(f"x : {x1 } , y : {y1} , x2: {x2} , y2: {y2} , type : {type(x1)} ")
                            center = int((x1+x2)/2) , int((y1+y2)/2)
                            
                            cv2.circle(im0 , (center) , 3 , (255,255,255) , 3) # 중심좌표 시각화
                            ## depth 정보 가져오기(depth 카메라 해상도 (640*480))
                            cv2.circle(depth_colormap , (center) , 3 , (255,255,255) , 3 )
                            distance = aligned_depth_frame.get_distance(center[0] , center[1])
                            print(distance)
                            ##
                            results['center'] = center
                            results['distance'] = distance
                # Stream results (webcam 화면 으로 부터받은 결과를 출력)
                im0 = annotator.result()
                if save_img:
                    hstack_img = np.hstack((origin_color_image , im0 , depth_colormap))
                    out.write(hstack_img) # 동영상 저장
            if view_img:
                cv2.imshow("original", origin_color_image)
                cv2.imshow(str("result"), im0)
                cv2.imshow("depth" ,depth_colormap)
                # cv2.waitKey(1)  # 1 millisecondqq
                
            # Print results
            end = time.time()-start
            print("inference time : " , end) # 1번 인식하는데 0.5초 정도 FPS:2 정도 나옴,
            t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if update:
                strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
            if cv2.waitKey(1) == ord('q'):
                break 
    finally:
        # Stop streaming
        if save_img:
            out.release()
        if save_img:
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
        if save_txt:
            LOGGER.info(f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else '') 
        pipeline.stop()
        cv2.destroyAllWindows()

        
        # 동영상 저장 코드 있어야 할 듯 
def parse_opt():
    parser = argparse.ArgumentParser() # --옵션인수지정 (- 1개만 적어도 옵션 설정가능하나 프로그램 실행시에는 문제 없으나 프로그램 내에서 -실행시 오류)
    # type : 기본적인 데이터는 str타입으로 저장되어서 정수형이나 실수형으로 지정할 경우 type지정도 가능함.
    # default : 옵션 지정을 안해줄시 자동으로 저장되는 값
    # action : 플래그로 사용할 수 있어서 실생시에 --옵션값이 지정되어 있는 경우 True , 그렇지 않은경우 False가 된다 구체적인 값을 지정하는 것이 아닌 
    #          True나 False를 지정하게 하고 싶은 경우 action 사용(stroe_false인 경우는 값이 지정시 False , 지정하지 않으면 True가 됨.)
    # help : 사용자가 지정한 옵션에 대한 설명 코멘트
    
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/best_0215+30.pt', help='model path or triton URL')
    # parser.add_argument('--source', type=str, default=ROOT / 'dataset/test_data/', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold') # confidence score가 일정값을 넘지 않는 것은 추론x
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold') # NMS 할때 IOU의 임계값 
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos') # 저장 여부 결정 옵션을 지정하면 true가 되어 저장을 하지 않음.
    parser.add_argument('--save', action='store_false', help='do not save images/videos') # 저장 여부 결정, 옵션을 지정하면 false가 되어 저장.
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args() # 위에서 정의한 옵션 인수들을 분석한 후 opt에 저장
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
