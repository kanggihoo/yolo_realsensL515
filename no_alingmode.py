import pyrealsense2 as rs
import numpy as np
import cv2


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

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
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        
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
        
        for color_point in color_points:
        # color에서의 픽셀좌표를 depth fream에서의 픽셀좌표로 변경 (rs.rs2_project_color_pixel_to_depth_pixel)
            
            depth_point_ = rs.rs2_project_color_pixel_to_depth_pixel( 
                            depth_frame.get_data(), depth_scale,
                            depth_min, depth_max,
                            depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, color_point)
            # print(depth_point_)
            cv2.circle(color_image , (int(color_point[0]) , int(color_point[1])) , radius =5 , color = (255,255,255) , thickness=-1)
            cv2.circle(depth_colormap , (int(depth_point_[0]) , int(depth_point_[1])) , 5, (0,0,0) , -1)
            
        
        cv2.imshow("color" , color_image)
        cv2.imshow("depth" , depth_colormap)
        if cv2.waitKey(1) ==ord('q'):
            break
        

finally:
    pipeline.stop()



