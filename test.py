import pyrealsense2 as rs
import numpy as np
import cv2
import copy
import math


class check:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30) # 640*480 , 1024*768
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # 640*360 , 640*480 , 960*540 , 1280*720 , 1920*1080
        self.profile = self.pipeline.start(config)
        
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        sensor_dep = self.profile.get_device().first_depth_sensor()
        sensor_dep.set_option(rs.option.min_distance , 0)
        sensor_dep.set_option(rs.option.visual_preset , 3)

        self.count = 0
        self.point = []
        
    def mouse_click(self , event , x,y,flag, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth = self.aligned_depth_frame.get_distance(x,y)
            depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [x, y], depth)
            print(f"x,y = {x,y},depth : {depth} , depth_point: {depth_point}")
            self.count += 1
            self.point.append(depth_point)
            if self.count % 2 ==0:
                distance = self.cal_distance(self.point)          
                self.count %= 2
                self.point = []
                print(distance)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.FOV()
            self.cm_per_pixel_ratio()
            
    def cal_distance(self , point):
        p1 = point[0]
        p2 = point[1]
        print("p1 : " , p1)
        print("p2 : " , p2)
        distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return distance
    
    # 이방법은 
    def FOV(self):
        left_top= (0,0) ; right_top = (639,479)
        left_bottom = (0,479) ; right_bottom = (639,479) 
        center = (320-1 , 240-1)
        
        left_top_depth = self.aligned_depth_frame.get_distance(left_top[0] , left_top[1])
        left_top_depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [left_top[0] , left_top[1]], left_top_depth)
        right_top_depth = self.aligned_depth_frame.get_distance(right_top[0] , right_top[1])
        right_top_depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [right_top[0] , right_top[1]], right_top_depth)
        
        left_bottom_depth = self.aligned_depth_frame.get_distance(left_bottom[0] , left_bottom[1])
        left_bottom_depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [left_bottom[0] , left_bottom[1]], left_bottom_depth)
        right_bottom_depth = self.aligned_depth_frame.get_distance(right_bottom[0] , right_bottom[1])
        right_bottom_depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [right_bottom[0] , right_bottom[1]], right_bottom_depth)
        
        center_depth = self.aligned_depth_frame.get_distance(center[0] , center[1])
        center_depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [center[0] , center[1]], center_depth)
        
        total_x = right_bottom_depth_point[0] - left_top_depth_point[0]
        total_y = right_bottom_depth_point[1] - left_top_depth_point[1]
        
        
        z_error = right_bottom_depth_point[2] - left_top_depth_point[2]
        
        half_x = right_bottom_depth_point[0] - center_depth_point[0]
        half_y = right_bottom_depth_point[1] - center_depth_point[1]
        z_error2 = right_bottom_depth_point[2] - center_depth_point[2]
        
        print(f"left_top : {left_top}")
        print(f"right_bottom : {right_bottom}")
        print(f"center : {center}")
        
        print(f"left_top_depth_point : {left_top_depth_point}")
        print(f"right_bottom_depth_point : {right_bottom_depth_point}")
        print(f"center_depth_point : {center_depth_point}")
        
        print(f"total_x = {total_x} , total_y = {total_y} , z_error = {z_error}")
        print(f"half_x = {half_x} , half_y = {half_y} , z_error2 = {z_error2}")
    def cm_per_pixel_ratio(self ,x = 319 , y = 239):
        
        # 1pixel 
        center_depth = self.aligned_depth_frame.get_distance(x , y)
        center_depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [x , y], center_depth)
        center_depth_point2 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [x+1 , y+1], center_depth)
        one_pixel2cm = (center_depth_point2[0] - center_depth_point[0])*100 , (center_depth_point2[1] - center_depth_point[1])*100
        print("1pixel x : " ,one_pixel2cm)
        
        # 10pixel 
        depth = self.aligned_depth_frame.get_distance(x-4 , y-4)
        depth2 = self.aligned_depth_frame.get_distance(x+5 , y+5)
        
        center_depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [x-4 , y-4], center_depth)
        center_depth_point2 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [x+5, y+5], center_depth)
        ten_pixel2cm = (center_depth_point2[0] - center_depth_point[0])*100 , (center_depth_point2[1] - center_depth_point[1])*100
        print("10pixel x : " ,ten_pixel2cm)
        
        # 100pixel 
        depth = self.aligned_depth_frame.get_distance(x-49 , y-49)
        depth2 = self.aligned_depth_frame.get_distance(x+50 , y+50)
        
        center_depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [x-49 , y-49], center_depth)
        center_depth_point2 = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [x+50, y+50], center_depth)
        ten2_pixel2cm = (center_depth_point2[0] - center_depth_point[0])*100 , (center_depth_point2[1] - center_depth_point[1])*100
        print("100pixel x : " ,ten2_pixel2cm)
        
        
        
        
        
        
        

    def video(self):
        while True:
            # Get frameset of color and depth
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            self.aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            self.color_frame = aligned_frames.get_color_frame()
            
            # # Validate that both frames are valid
            if not self.aligned_depth_frame or not self.color_frame:
                continue

            depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
           
            color_image = np.asanyarray(self.color_frame.get_data())
            
            self.depth_intrin = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            print("depth_intrin : {}".format(self.depth_intrin))
            print(self.color_frame.profile.as_video_stream_profile().intrinsics)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
 
            self.img = color_image
            self.copy_img = color_image.copy()
            cv2.namedWindow("color_stream" , cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('color_stream' , self.mouse_click)
            cv2.circle(self.img , (int(640/2)-1 , int(480/2)-1) , 3, (255,255,255) , -1)
            cv2.circle(self.img , (0,0) , 3, (255,255,255) , -1)
            cv2.line(self.img , (0,239),(639,239) , (255,0,0),1)
            cv2.imshow("color_stream" , self.img)
            cv2.imshow("depth_image" , depth_colormap)
            
            
            key = cv2.waitKey(1)
            # # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        
        
        
                
if __name__ == "__main__":
    check().video()
    
    