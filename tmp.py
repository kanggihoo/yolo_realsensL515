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
        
        self.pixel2cm = 0.159514 # 약 1pixel = 0.16cm 
        self.cm2pixel = 6.269057 # 약 1cm = 6.27pixel 
        self.SetCameraConfig()
        self.GetCameraConfig()
        
        self.color_points = self.make_arr()

    
    def camera_config(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # 640*480 , 1024*768
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 640*360 , 640*480 , 960*540 , 1280*720 , 1920*1080
        self.profile = self.pipeline.start(config)
    def make_arr(self):
        # color_roi = (380,280)
        # x = np.array([ list(range(color_roi[0])) for i in range(color_roi[1]) ]).ravel()
        # y = np.array([ list(range(color_roi[0])) for i in range(color_roi[1]) ]).T.ravel()
        # x = x.reshape(-1,1)
        # y = y.reshape(-1,1)
        # new = np.concatenate((x,y) , axis = 1)
        # new[:,0] += 130
        # new[:,1] += 100
        new = [644,700]
        return new
            
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
                                
                
                for color_point in self.color_points:
                # color에서의 픽셀좌표를 depth fream에서의 픽셀좌표로 변경 (rs.rs2_project_color_pixel_to_depth_pixel)
                    print(color_point)
                    
                    depth_point_ = self.project_color_pixel_to_depth_pixel(color_point , depth_frame)
                    
                        
                
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
                #     } 
                if cv2.waitKey(1) ==ord('q'):
                    break        
                cv2.imshow("color" , color_image)
                cv2.imshow("depth" , depth_colormap)
                

        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()
            

        


camera = Checkscale()
center = (int(640/2) , int(480/2))
camera.show()

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
#         depth_point_array[y-100,x-130] = depth_point[2]
#         x += 1
#         if x == 510:
#             print(depth[y-100,x-131] , depth[y-100,0])
#             y +=1
#             x =130
#             print(y)
#             frames = camera.pipeline.wait_for_frames()
#             depth_frame = frames.get_depth_frame()
            
            
#         if y == 380:
#             break
        
       
# finally :
#     print("complete")
#     # # np.save(r"C:\Users\11kkh\Desktop\yolov5\depth_array_noalign2.npy" , depth)
#     # np.save(r"C:\Users\11kkh\Desktop\yolov5\depth_array_noalign3.npy" , depth_point_array)
        
        
        
