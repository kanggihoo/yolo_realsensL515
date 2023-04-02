# import qrcode
import cv2
from pathlib import Path
import os
import sys
import json
import numpy as np
from QR_code_function import detect_qr_video , detect_qr 


FILE = Path(__file__).parent
sys.path.append(str(FILE))
qr_name = 'test.jpg'
QR_PATH = FILE / qr_name

ROOT = FILE.parents[0]
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
path = Path(os.path.relpath(ROOT, Path.cwd()))

# QR 코드 생성

# qr_img1 = make_qrcode(size = 10 , border =2 , weight= 10 , height= 15)
# qr_img2 = make_qrcode(size = 15 , border =2 , weight= 40 , height= 45)
# qr_img3 = make_qrcode(size = 10 , border =2 ,error=qrcode.constants.ERROR_CORRECT_H,  weight= 10 , height= 15)
# qr_img4 = make_qrcode(size = 5 , border =5 , weight= 50 , height= 55)
# qr_img5 = make_qrcode(size = 5 , border =5 , weight= 60 , height= 66)
# save_img(qr_img1 , file_name='qr_img1'+"2.jpg")
# save_img(qr_img2 , file_name='qr_img2'+"2.jpg")
# save_img(qr_img3 , file_name='qr_img3'+"2.jpg")
# save_img(qr_img4 , file_name='qr_img4'+"2.jpg")
# save_img(qr_img5 , file_name='qr_img5'+"2.jpg")

# # 이미지 읽기
# img = cv2.imread(str(FILE / 'qr_test2.jpg'))
# img = cv2.resize(img , (640,640))
# data , rects = detect_qr(img)
# gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

# range = 40
# for rect in rects:
#     x,y,w,h = rect[0]-range , rect[1]-range , rect[2]+range*2, rect[3]+range*2
#     cv2.rectangle(img , (x, y) , (x+w ,y+h) , (255,0,0) , 2)
#     qr_img = img[y:y+h ,x:x+w]
#     gray = cv2.cvtColor(qr_img , cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray , ksize=(5,5) , sigmaX=0)
#     canny = cv2.Canny(gray , 20 , 200 )
#     contours , _ = cv2.findContours(canny , mode = cv2.RETR_LIST , method=cv2.CHAIN_APPROX_SIMPLE)
#     tmp_result = np.zeros_like( qr_img,dtype=np.uint8)
#     # cv2.drawContours(tmp_result, contours , contourIdx= -1 , color = (0,0,255) , thickness=2)
#     contour_dict = []
    
#     # MIN_AREA , MAX_AREA = 10000 , tmp_result.shape[0] * tmp_result.shape[1]
#     # MIN_WIDTH , MIN_HEIGHT = 100,100 
    
#     # idx = 0
#     # for contour in contours:
#     #     x,y,w,h = cv2.boundingRect(contour)
#     #     if w*h > MIN_AREA and w*h < MAX_AREA and w > MIN_HEIGHT and h > MIN_HEIGHT:
#     #         contour_dict.append({
#     #             "contour" : contour,
#     #             "x": x,
#     #             "y":y,
#     #             "w":w,
#     #             "h":h,
#     #             "cx": x+(w/2),
#     #             "cy": y+(h/2),
#     #             "idx" : idx
#     #         })
#     #         idx += 1
#     #         # cv2.rectangle(tmp_result , (x,y) , (x+w , y+h) , color = (255,255,255) , thickness=2)
#     #         rectangle = cv2.minAreaRect(contour)
#     #         box = cv2.boxPoints(rectangle)
#     #         box = np.intp(box)
#     #         print(box , idx)
#     #         cv2.drawContours(tmp_result , [box] , -1 , (255,255,255) , 2)
#     # tmp_result2 = np.zeros_like(qr_img , dtype=np.uint8)
#     # for contour in contour_dict:
#     #     cv2.drawContours(tmp_result2 , contour['contour'] , contourIdx= -1 , color = (255,255,255) , thickness=2)
#     #     print(len(contour['contour']))
    

# cv2.imshow("qr" , img)
# cv2.imshow("canny" , canny)
# cv2.imshow("gray" ,gray)
# cv2.imshow('qr_img' , qr_img)
# cv2.imshow('tmp_result' , tmp_result)
# # cv2.imshow('tmp_result2' , tmp_result2)
# cv2.waitKey()


# 동영상 스트리밍(노트북 캠)
# cap = cv2.VideoCapture(0)

# while (True):
#     ret , frame = cap.read()
#     if not ret:
#         break
#     tmp = frame.copy()
#     tmp = tmp[ 100:150 , 100:150]
#     upsacle = cv2.resize(tmp , (300,300) , interpolation=cv2.INTER_LINEAR)
#     upsacle = cv2.cvtColor(upsacle , cv2.COLOR_BGR2GRAY)
#     _,binary = cv2.threshold(upsacle , 100 , 255 , cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        
#     qr_frame , data = detect_qr_video(upsacle)
    
#     cv2.rectangle(frame , (100,100) , (150,150) , (255,0,0) , 2)
#     cv2.imshow('video' , frame)
#     cv2.imshow('video2' , tmp)
#     cv2.imshow('upsacle' , upsacle)
#     cv2.imshow('binary' , binary)
#     cv2.imshow('video4' , qr_frame)
    
#     if cv2.waitKey(33) == ord('q'):
#         break

# cv2.destroyAllWindows()


# 동영상 스트리밍(intelrealsense)
############################################################################################################################################
import pyrealsense2 as rs
import cv2


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 640*480 , 1024*768
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30) # 640*360 , 640*480 , 960*540 , 1280*720 , 1920*1080
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)


sensor_dep = profile.get_device().first_depth_sensor()
print("min_distance : ",sensor_dep.get_option(rs.option.min_distance))
sensor_dep.set_option(rs.option.min_distance , 0)
print("set_min_distance : ",sensor_dep.get_option(rs.option.min_distance))

print("visual_preset : ",sensor_dep.get_option(rs.option.visual_preset))
sensor_dep.set_option(rs.option.visual_preset , 3)
print("set_min_visual_preset : ",sensor_dep.get_option(rs.option.visual_preset))

save_path = str(path / "resultsd")
save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
print(save_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(save_path, fourcc, 15, (640, 480)) #  fps, w, h = 30, im0.shape[1], im0.shape[0]

try:
    while 1:
        
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # # frames.get_depth_frame() is a 640x360 depth image

        # # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        center = int(1920/2),int(1080/2)
        xy_depth = aligned_depth_frame.get_distance(center[0], center[1])
        xy_depth = round(xy_depth , 2)
        
        import time
        start = time.time()
        img = cv2.resize(color_image , (640,480))
        end = time.time() - start
        print('걸린시간 : ' , end )
        
       
    

#######################################################################################################
        # tmp = color_image.copy()
        # tmp = tmp[ 100:200+1 , 100:200+1]
        # upsacle = cv2.resize(tmp , (300,300) , interpolation=cv2.INTER_LINEAR)
        # upsacle = cv2.cvtColor(upsacle , cv2.COLOR_BGR2GRAY)
        # _,binary = cv2.threshold(upsacle , 100 , 255 , cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # qr_frame , data = detect_qr_video(binary)
        

        qr_frame , data  = detect_qr_video(color_image)
        cv2.putText(qr_frame , f"{str(xy_depth)}m" , (500 , 100) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0,0,0))
        print(len(data))
        print('*'*20)
        
        # for i in data:
        #     print(i['weight'])
            
        # cv2.rectangle(color_image , (100,100) , (200,200) , (255,255,255) , 2)
        # cv2.imshow('video' , color_image)
        # cv2.imshow('video2' , tmp)
        # cv2.imshow('upsacle' , upsacle)
        # cv2.imshow('binary' , binary)
        out.write(qr_frame)
        cv2.imshow('video4' , qr_frame)
        if cv2.waitKey(33) == ord('q'):
            break    

finally:
    cv2.destroyAllWindows()
    pipeline.stop()
    out.release()


