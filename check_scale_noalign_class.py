import pyrealsense2 as rs
import numpy as np
import cv2

class Checkscale():
    def __init__(self):
        self.camera_config()
        
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()    
        self.mouse_position = []
        
        self.depth_min = 0.11 #meter
        self.depth_max = 1.0 #meter

        self.depth_intrin = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics() # depth 카메라의 내부 파라미터
        self.color_intrin = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics() # color 카메라의 내부 파라미터

        self.depth_to_color_extrin =  self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( self.profile.get_stream(rs.stream.color)) # depth => color 외부파라미터
        self.color_to_depth_extrin =  self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( self.profile.get_stream(rs.stream.depth)) # color => depth 외부 파라미터
        
        self.print_flag = None

            
    def camera_config(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)
    
    def mouse_callback(self , event , x,y,flags , param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param.append((x,y))
            self.print_flag = 1
        if event == cv2.EVENT_RBUTTONDOWN: # 마지막 리스트 값 삭제
            self.mouse_position.pop()
            
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
        sensor_dep = camera.profile.get_device().first_depth_sensor()
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
                
                
            
                color_points = [
                    [400.0, 150.0],
                    [560.0, 150.0],
                    [560.0, 260.0],
                    [400.0, 260.0]
                ]                     
                
                for color_point in color_points:
                # color에서의 픽셀좌표를 depth fream에서의 픽셀좌표로 변경 (rs.rs2_project_color_pixel_to_depth_pixel)
                    
                    depth_point_ = self.project_color_pixel_to_depth_pixel(color_point , depth_frame)
                    cv2.circle(color_image , (int(color_point[0]) , int(color_point[1])) , radius =5 , color = (255,255,255) , thickness=-1)
                    cv2.circle(depth_colormap , (int(depth_point_[0]) , int(depth_point_[1])) , 5, (0,0,0) , -1)
                    
                
                
                if self.mouse_position != None:
                    for position in self.mouse_position:
                        # 마우스 좌표 시각화
                        cv2.circle(color_image , (position[0] , position[1]) , 3 , (255,255,255) , -1)
                        cv2.putText(color_image , f"[{str(position[0])} , {str(position[1])}]" , (position[0]-50,position[1]+20) ,cv2.FONT_ITALIC,0.4,(255,0,0),2)
                        
                        # depth_colormap 시각화
                        depth_pixel = self.project_color_pixel_to_depth_pixel(position , depth_frame)
                        depth_pixel_round = list(map(lambda x : int(round(x,1)) , depth_pixel))
                        cv2.circle(depth_colormap, (int(depth_pixel_round[0]) , int(depth_pixel_round[1])) , 3 , (255,255,255) , -1)
                    
                        if self.print_flag is not None:
                            # deprojection
                            depth , depth_point = self.DeProjectDepthPixeltoDepthPoint(depth_pixel[0] , depth_pixel[1] , depth_frame)
                            print(f"(x,y) : {position[0] , position[1]} , depth = {depth} , depth_point = {depth_point}")
                            
                    self.print_flag = None
                        
                        
                        
                
                if cv2.waitKey(1) ==ord('q'):
                    break
                
                cv2.imshow("color" , color_image)
                cv2.imshow("depth" , depth_colormap)
                cv2.setMouseCallback('color' , self.mouse_callback , param=  self.mouse_position)
                

        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()



camera = Checkscale()
center = (int(640/2) , int(480/2))
camera.GetCameraConfig()



frames = camera.pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()

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
