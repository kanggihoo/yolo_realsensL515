import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm

class Checkscale():
    def __init__(self):
        self.camera_config()
        
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()    
        self.mouse_position = []
        
        self.depth_min = 0.1 #meter
        self.depth_max = 2.0 #meter

        self.depth_intrin = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics() # depth 카메라의 내부 파라미터
        self.color_intrin = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics() # color 카메라의 내부 파라미터

        self.depth_to_color_extrin =  self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( self.profile.get_stream(rs.stream.color)) # depth => color 외부파라미터
        self.color_to_depth_extrin =  self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( self.profile.get_stream(rs.stream.depth)) # color => depth 외부 파라미터
        
        self.print_flag = None
        self.pre_mouse = None
        # self.df = pd.DataFrame(columns = ['cur_pixel' , 'pre_pixel' , 'dx_pixel' , 'dy_pixel' , 'dx_point' , 'dy_point' , 'dz_point' ,
        #                                   'cm_per_pixel(x)', 'pixel_per_cm(x)', 'cm_per_pixel(y)', 'pixel_per_cm(y)'])
        self.df = pd.DataFrame(columns = ['cur_pixel' , 'pre_pixel' , 'dx_pixel' , 'dy_pixel' , 'dx_point' , 'dy_point' , 'dz_point' ])
        
        self.SetCameraConfig()
        self.GetCameraConfig()
        print(f"depth_intrin : {self.depth_intrin}")
        
        print(f"color_intrin : {self.color_intrin}")
        print(f"depth_to_color_extrin : {self.depth_to_color_extrin}")
        print(f"color_to_depth_extrin : {self.color_to_depth_extrin}")

    def Drawline_pixel(self , image , start:int , num_pixel:int , direction = 'x' ):
        '''
        앞에서 구한 pixel2cm에 맞게 직선을 그려주는 함수
        # input : numpy array 배열
        # start : 직선의 시작 위치 
        # direction : 'x' or 'y' 빙향
        # num_pixel : self.pixel2cm의 몇 배를 할 것인지 
        # output : line을 그린 후 numpy array 반환 
        '''
        
        
        # 1.6cm => 10pixel 
        start_x , start_y = start[0] , start[1]
        if direction =='x':
            end_x , end_y = start_x+num_pixel , start_y
        else:
            end_x , end_y = start_x , start_y+num_pixel
            
        cv2.line(image , (start_x , start_y) , (end_x , end_y) , (255,255,255) , 2 )
        cv2.putText(image , f"{num_pixel}pixel = {round(self.pixel2cm*num_pixel,3)}cm" , (start_x , start_y-20) , cv2.FONT_ITALIC,0.5,(0,0,0),2 )
        return image
    def Drawline_cm(self , image , start:int , cm:float , direction = 'x' ): # 특정 cm 만큼 직선을 그려주는 함수
        '''
        앞에서 구한 pixel2cm에 맞게 직선을 그려주는 함수
        # input : numpy array 배열
        # start : 직선의 시작 위치 
        # direction : 'x' or 'y' 빙향
        # cm: 원하는 cm 
        # output : line을 그린 후 numpy array 반환 
        '''
        
        pixel = cm*self.cm2pixel
        pixel_round = int(round(pixel,0))
        start_x , start_y = start[0] , start[1]
        if direction =='x':
            end_x , end_y = start_x+pixel_round , start_y
        else:
            end_x , end_y = start_x , start_y+pixel_round
            
        cv2.line(image , (start_x , start_y) , (end_x , end_y) , (255,255,255) , 2 )
        cv2.putText(image , f"{cm}cm = {pixel_round}pixel ({pixel})" , (start_x , start_y-20) , cv2.FONT_ITALIC,0.5,(0,0,0),2 )
        return image
    
    def camera_config(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 640*480 , 1024*768
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 640*360 , 640*480 , 960*540 , 1280*720 , 1920*1080
        self.profile = self.pipeline.start(config)
    
    def mouse_callback(self , event , x,y,flags , param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(param) !=0:
                last = param[-1]
                # print(f"pixel dx = {abs(last[0] - x)} , pixel dy = {abs(last[1] - y)}")
        
            param.append((x,y))
            self.print_flag = 1
        if event == cv2.EVENT_RBUTTONDOWN: # 마지막 리스트 값 삭제
            self.mouse_position.pop()
            self.df = self.df[:-1]
            
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
        
        # [x_depth_pixel , y_depth_pixel]여기 부분은 int형이 안들어가도 되지 않나? => int형 안하면 오류 발생하는지 확인 해보기
        return depth, depth_point
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

    def show(self):
        try:
            while 1:
                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                
                # # Get aligned frames
            
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.circle(color_image , (320,117) , 3 , (255,255,255) , 3)
                
    
                # color_points = [
                #     [400.0, 150.0],
                #     [560.0, 150.0],
                #     [560.0, 260.0],
                #     [400.0, 260.0]
                # ]                     
                
                # for color_point in color_points:
                # # color에서의 픽셀좌표를 depth fream에서의 픽셀좌표로 변경 (rs.rs2_project_color_pixel_to_depth_pixel)
                    
                #     depth_point_ = self.project_color_pixel_to_depth_pixel(color_point , depth_frame)
                #     cv2.circle(color_image , (int(color_point[0]) , int(color_point[1])) , radius =5 , color = (255,255,255) , thickness=-1)
                #     cv2.circle(depth_colormap , (int(depth_point_[0]) , int(depth_point_[1])) , 5, (0,0,0) , -1)
                    
                
                
                if self.mouse_position != None:
                    for position in self.mouse_position:
                        # 마우스 좌표 시각화
                        cv2.circle(color_image , (position[0] , position[1]) , 3 , (255,255,255) , -1)
                        # cv2.putText(color_image , f"[{str(position[0])} , {str(position[1])}]" , (position[0]-50,position[1]+20) ,cv2.FONT_ITALIC,0.4,(255,0,0),2)
                        
                        # depth_colormap 시각화
                        depth_pixel = self.project_color_pixel_to_depth_pixel(position , depth_frame)
                        depth_pixel_tmp = self.project_color_pixel_to_depth_pixel((134,341) , depth_frame)
                        depth_pixel_round = list(map(lambda x : int(round(x,0)) , depth_pixel))
                        cv2.circle(depth_colormap, (int(depth_pixel_round[0]) , int(depth_pixel_round[1])) , 3 , (255,255,255) , -1)
                        
                        depth , depth_point = self.DeProjectDepthPixeltoDepthPoint(depth_pixel[0] , depth_pixel[1] , depth_frame)
                        
                        depth_point_round = list(map(lambda x: round(x*100,3) ,  depth_point) )
                        
                        cv2.putText(color_image , f"{str(round(depth_image[depth_pixel_round[1] , depth_pixel_round[0]]*self.depth_scale,3))}" , (position[0]-30,position[1]+10) ,cv2.FONT_ITALIC,0.5,(255,0,0),2)
                        cv2.putText(color_image , f"{str(depth_point_round)}" , (position[0]-80,position[1]+30) ,cv2.FONT_ITALIC,0.5,(255,0,0),2)
                        
                
                # if len(self.mouse_position) >= 2:
                #     cur = self.mouse_position[-1]
                #     pre = self.mouse_position[-2]
                #     cur_depth_pixel = self.project_color_pixel_to_depth_pixel(cur , depth_frame)
                #     pre_depth_pixel = self.project_color_pixel_to_depth_pixel(pre , depth_frame)
                #     # deprojection
                #     cur_depth , cur_depth_point = self.DeProjectDepthPixeltoDepthPoint(cur_depth_pixel[0] , cur_depth_pixel[1] , depth_frame)
                #     pre_depth , pre_depth_point = self.DeProjectDepthPixeltoDepthPoint(pre_depth_pixel[0] , pre_depth_pixel[1] , depth_frame)
                    
                #     result_dict={
                #         'cur_pixel': (cur[0],cur[1]) ,
                #         'pre_pixel' : (pre[0],pre[1]),
                #         'dx_pixel' : abs(cur[0]-pre[0]), 
                #         'dy_pixel' : abs(cur[1]-pre[1]) ,
                #         'dx_point' : abs(cur_depth_point[0] - pre_depth_point[0]*100),
                #         'dy_point' : abs(cur_depth_point[1] - pre_depth_point[1]*100),
                #         'dz_point' : abs(cur_depth_point[2] - pre_depth_point[2]*100),
                #         # 'cm_per_pixel(x)' : abs(cur[0]-pre[0]) / (abs(cur_depth_point[0] - pre_depth_point[0]*100)),
                #         # 'pixel_per_cm(x)' :  abs(cur_depth_point[0] - pre_depth_point[0])*100 / abs(cur[0]-pre[0]),
                #         # 'cm_per_pixel(y)' : abs(cur[1]-pre[1]) / (abs(cur_depth_point[1] - pre_depth_point[1])*100),
                #         # 'pixel_per_cm(y)' :  abs(cur_depth_point[1] - pre_depth_point[1])*100 / abs(cur[1]-pre[1])
                #         }
                            
                #     if self.print_flag is not None:
                #         print("result!","*"*20)
                #         print(f"cur = {cur[0] , cur[1]} , pre = {pre[0] , pre[1]}")
                #         print(f"cur_depth = {depth_image[cur[1] , cur[0]]*self.depth_scale} , pre_depth = {depth_image[pre[1] , pre[0]]*self.depth_scale}")
                #         print(f"cur_depth2 = {cur_depth} , pre_depth2 = {pre_depth}")
                #         print(f"cur_depth_point = {cur_depth_point} , pre_depth_point = {pre_depth_point}")
                        
                #         print(f"dx = {abs(cur_depth_point[0] - pre_depth_point[0])*100}")
                #         print(f"dy = {abs(cur_depth_point[1] - pre_depth_point[1])*100}")
                #         print(f"dz = {abs(cur_depth_point[2] - pre_depth_point[2])*100}")
                #         print("*"*20)
                        
                #         self.df.loc[len(self.df)] = result_dict.values()
                #         # result_dict={
                #         #  'cur_pixel': (cur[0],cur[1]) ,
                #         #  'pre_pixel' : (pre[0],pre[1]),
                #         #  'dx_pixel' : abs(cur[0]-pre[0]), 
                #         #  'dy_pixel' : abs(cur[1]-pre[1]) ,
                #         #  'dx_point' : abs(cur_depth_point[0] - pre_depth_point[0]),
                #         #  'dy_point' : abs(cur_depth_point[1] - pre_depth_point[1]),
                #         #  'dz_point' : abs(cur_depth_point[2] - pre_depth_point[2]),
                #         #  'cm_per_pixel' : abs(cur[0]-pre[0]) / abs(cur_depth_point[0] - pre_depth_point[0]),
                #         #  'pixel_per_cm' :  abs(cur_depth_point[0] - pre_depth_point[0]) / abs(cur[0]-pre[0])}
                #         # self.df = self.df.append(result_dict , ignore_index = True) 
                #         self.print_flag = None
                        
                        
                if len(self.mouse_position) >= 1:
                    cur = self.mouse_position[-1]
                    
                    cur_depth_pixel = self.project_color_pixel_to_depth_pixel(cur , depth_frame)
                    
                    # deprojection
                    cur_depth , cur_depth_point = self.DeProjectDepthPixeltoDepthPoint(cur_depth_pixel[0] , cur_depth_pixel[1] , depth_frame)
                            
                    if self.print_flag is not None:
                        print("result!","*"*20)
                        print(f"cur_color_pixel = {cur[0] , cur[1]}")
                        print(f"cur_depth_pixel = {cur_depth_pixel}")
                        print(f"cur_depth = {depth_image[cur[1] , cur[0]]*self.depth_scale}")
                        print(f"depth_point = {cur_depth} ")
                        print(f"cur_depth_point = {cur_depth_point} ")
                        
                        self.print_flag = None
                        
                if cv2.waitKey(1) ==ord('q'):
                    break
                # 이미지 저장
                if cv2.waitKey(1) == ord('s'):
                    dirpath = Path(__file__).resolve().parents[0]
                    image_name = Path('result').with_suffix('.jpg')
                    file_name = str(dirpath / image_name)
                    cv2.imwrite(file_name , color_image)
                    print("image_save")
                
                cv2.imshow("color" , color_image)
                cv2.imshow("depth" , depth_colormap)
                cv2.setMouseCallback('color' , self.mouse_callback , param=  self.mouse_position)
                

        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()
            
    def show_line(self):
        try:
            while 1:
                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                
                # # Get aligned frames
            
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                # self.Drawline_pixel(color_image , (int(640/2) , int(480/2)) , 100 , 'x' )
                # self.Drawline_pixel(color_image , (int(640/2) , int(480/2)) , 100 , 'y' )
                
                self.Drawline_cm(color_image , (int(640/2) , int(480/2)) , 12.0 , 'x')
                self.Drawline_cm(color_image , (int(640/2) , int(480/2)) , 12.0 , 'y')
            
                        
                if cv2.waitKey(1) ==ord('q'):
                    break
                
                # 이미지 저장
                if cv2.waitKey(1) == ord('s'):
                    dirpath = Path(__file__).resolve().parents[0]
                    image_name = Path('result').with_suffix('.jpg')
                    file_name = str(dirpath / image_name)
                    cv2.imwrite(file_name , color_image)
                    print("image_save")
                
                cv2.imshow("color" , color_image)
                cv2.imshow("depth" , depth_colormap)
                

        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()
        


camera = Checkscale()
center = (int(640/2) , int(480/2))
# camera.GetCameraConfig() 
# sclae 얻기 ------------------------------------------------------------------------------
camera.show()
# camera.df.to_csv(r"C:\Users\11kkh\Desktop\yolov5\check_scale.csv", index = False)

# scale 결과 값 출력 ------------------------------------------------------------------------------
# camera.show_line()                                    



# #depth 값 저장 ---------------------------------------
frames = camera.pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
depth_scale = camera.depth_scale
# time.sleep(3)
# depth_image = np.round(np.asanyarray(depth_frame.get_data()) * depth_scale,3)

##  depth = depth_frame.get_distance(int(x_depth_pixel), int(y_depth_pixel))와 단순히 depth_frame.get_data()depth_scale 곱의 차이?
# # import random

# # for i in range(10):
# #     rand1 = random.randint(0,480)
# #     rand2 = random.randint(0,640)
# #     print(rand1 , rand2)
# #     print(depth_image[rand1,rand2] , depth_frame.get_distance(rand2,rand1))

# np.save(r"C:\Users\11kkh\Desktop\yolov5\depth_array_noalign.npy" , depth_image)
# print("save 완료!!")

# depth =  np.load(r'C:\Users\11kkh\Desktop\yolov5\depth_array_noalign2.npy') # (280x380) # depth_image에서의 depth 값을 저장하는 배열
# depth_point_array =  np.load(r'C:\Users\11kkh\Desktop\yolov5\depth_array_noalign3.npy')# (280x380) # color_image의 픽셀=> depth_image 픽셀변환 후 그때의 3D좌표의 dz값을 저장하는 배열
# y = 100 ; x = 130
# frames = camera.pipeline.wait_for_frames()
# depth_frame = frames.get_depth_frame()
# try:
#     while 1:
#         if not depth_frame :
#                 continue
#         depth_pixel = camera.project_color_pixel_to_depth_pixel( (x,y), depth_frame)
#         _, depth_point = camera.DeProjectDepthPixeltoDepthPoint(depth_pixel[0] , depth_pixel[1] , depth_frame)
#         depth_point_array[y-100,x-130] = round(depth_point[2],4)
#         x += 1
#         if x == 510:
#             # print(depth[y-100,x-131] , depth[y-100,0])
#             y +=1
#             x =130
#             print(y)
#             frames = camera.pipeline.wait_for_frames()
#             depth_frame = frames.get_depth_frame()
            
            
#         if y == 178 or y==258 or y ==349 or y==354:
#             y+=1
#             continue
#         if y == 380:
#             break
        
       
# finally :
#     print("complete")
#     # np.save(r"C:\Users\11kkh\Desktop\yolov5\depth_array_noalign2.npy" , depth)
#     np.save(r"C:\Users\11kkh\Desktop\yolov5\depth_array_noalign3.npy" , depth_point_array)
        
        
        

# 3D 좌표계에서 x,y가 0,0에 근접한 점 찾기
# Z = np.zeros(shape = (480,640))
# for y in range(480):
#     for x in range(640):
#         pixel = (x,y)
#         depth_pixel = camera.project_color_pixel_to_depth_pixel(pixel , depth_frame)
#         depth = depth_frame.get_distance(int(depth_pixel[0]), int(depth_pixel[1])) # depth 카메라 상의 픽셀 정보를 바탕으로 depth 갚 구함
#         Z[y,x] = depth
#     if y % 15==0:
#         print(y)
        
    
# print(Z.shape)
# np.save(r"C:\Users\11kkh\Desktop\yolov5\depth_array_noalign.npy" , Z)
# 
# color, depth resolution = (640,480)일때 렌즈 중심이 color 픽셀상에서 ( x = 331 , y = 269 )
