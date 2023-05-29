import pyrealsense2 as rs
import numpy as np
import cv2


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

def DeProjectDepthPixeltoDepthPoint(x_depth_pixel, y_depth_pixel , depth_frame , depth_intirin) -> float:
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
        depth = depth_frame.get_distance(int(round(x_depth_pixel,0)), int(round(y_depth_pixel,0))) # depth 카메라 상의 픽셀 정보를 바탕으로 depth 갚 구함
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(x_depth_pixel), int(y_depth_pixel)], depth) # depth 카메라의 픽셀과 depth 값을 통해 3D 좌표계 구함. 
        return depth, depth_point
    
try:
    while 1:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # # Get aligned frames
    
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.04), cv2.COLORMAP_HSV)
        
        
        depth_min = 0.11 #meter
        depth_max = 1.0 #meter

        depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics() # depth 카메라의 내부 파라미터
        color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics() # color 카메라의 내부 파라미터

        depth_to_color_extrin =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.color)) # depth => color 외부파라미터
        color_to_depth_extrin =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.depth)) # color => depth 외부 파라미터
        
        # print("")
        # print(f"depth_to_color_extrin : {depth_to_color_extrin}")
        # print("*"*20)
        # print(f"color_to_depth_extrin : {color_to_depth_extrin}")
        # color frame의 임의의 픽셀좌표
        color_points = [
            [400.0, 150.0],
            [560.0, 150.0],
            [560.0, 260.0],
            [400.0, 260.0]
        ]             
   
        depth_to_color_point = []
        depth_to_color_pixel = []
        color_to_depth_point = []
        
        for color_point in color_points:
        # color에서의 픽셀좌표를 depth fream에서의 픽셀좌표로 변경 (rs.rs2_project_color_pixel_to_depth_pixel)
            
            depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel( 
                            depth_frame.get_data(), depth_scale,
                            depth_min, depth_max,
                            depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, color_point)
            # print(depth_pixel)
            cv2.circle(color_image , (int(color_point[0]) , int(color_point[1])) , radius =5 , color = (255,255,255) , thickness=-1)
            cv2.circle(depth_colormap , (int(depth_pixel[0]) , int(depth_pixel[1])) , 5, (0,0,0) , -1)
            
            _ , depth_point = DeProjectDepthPixeltoDepthPoint(depth_pixel[0], depth_pixel[1] , depth_frame , depth_intrin)
            color_to_depth_point.append(depth_point)
            color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
            depth_to_color_point.append(color_point)
            color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)
            depth_to_color_pixel.append(color_pixel)
        
        a1 = depth_to_color_point[0] ; b1 = depth_to_color_point[1]
        print(f"{a1[0]-b1[0]} // {a1[1]-b1[1]} // {a1[2]-b1[2]}")
        
        a2 = depth_to_color_point[0] ; b2 = depth_to_color_point[1]
        print(f"{a2[0]-b2[0]} // {a2[1]-b2[1]} // {a2[2]-b2[2]}")
        
                
        
        cv2.imshow("color" , color_image)
        # cv2.imshow("color_copy" , color_image2)
        cv2.imshow("depth" , depth_colormap)
        if cv2.waitKey(1) ==ord('q'):
            break
        

finally:
    pipeline.stop()



