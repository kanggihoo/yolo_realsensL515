import pyrealsense2 as rs
import numpy as np
import cv2
import keyboard
import time



# depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# def DeProjectDepthPixeltoDepthPoint(x_depth_pixel, y_depth_pixel , depth_frame , depth_intirin) -> float:
#         '''
#         depth_frame에서의 pixel좌표를 바탕으로 3D 좌표계로 변환해줌. 
#         ## input 
#         x_depth_pixel : depth_image 상의 임의의 x 픽셀값
#         y_depth_pixel : depth_image 상의 임의의 y 픽셀값
#         depth_frame = depth_frame
        
#         ## return
#         depth : depth_image의 (x,y)에서 카메라와 떨어진 거리 (m)
#         depth_point : 카메라의 주점을 기준으로 하여 떨어진 거리 (x,y,z)m => 음수가 나올 수도 있음.
#         '''
#         depth = depth_frame.get_distance(int(round(x_depth_pixel,0)), int(round(y_depth_pixel,0))) # depth 카메라 상의 픽셀 정보를 바탕으로 depth 갚 구함
#         depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(x_depth_pixel), int(y_depth_pixel)], depth) # depth 카메라의 픽셀과 depth 값을 통해 3D 좌표계 구함. 
#         return depth, depth_point
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
while 1:
    key = keyboard.read_key()
    if key == 'p':
        print("p 방향키 누름")
    
        try:
                start = time.time()
                # Get frameset of color and depth
                profile = pipeline.start(config)
                while 1:
                    frames = pipeline.wait_for_frames()
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not depth_frame or not color_frame:
                        print("ssss")
                        continue
                    else :
                        break
                # # Get aligned frames
            
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.04), cv2.COLORMAP_HSV)
                # pipeline.stop()
                
                depth_min = 0.11 #meter
                depth_max = 1.0 #meter

                depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics() # depth 카메라의 내부 파라미터
                color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics() # color 카메라의 내부 파라미터

                depth_to_color_extrin =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.color)) # depth => color 외부파라미터
                color_to_depth_extrin =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.depth)) # color => depth 외부 파라미터
                
                print("total time : " , time.time()-start)
                while 1:
                    cv2.imshow("color" , color_image)
                    # cv2.imshow("color_copy" , color_image2)
                    cv2.imshow("depth" , depth_colormap)
                    if cv2.waitKey(1) ==ord('q'):
                        break
                

        finally:
            cv2.destroyAllWindows()
            



