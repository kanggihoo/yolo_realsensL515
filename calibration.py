import pyrealsense2 as rs
import numpy as np
import cv2
import time

# 기본 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 파이프라인 시작
profile = pipeline.start(config)

# 내부 파라미터 얻기
intrinsics_color = None
intrinsics_depth = None
# while 1:
#     # color, depth 프레임 수신
#     frames = pipeline.wait_for_frames()
#     color_frame = frames.get_color_frame()
#     depth_frame = frames.get_depth_frame()
#     if not color_frame or not depth_frame:
#         continue

#     # 내부 파라미터 얻기
#     intrinsics_color = color_frame.profile.as_video_stream_profile().intrinsics
#     intrinsics_depth = depth_frame.profile.as_video_stream_profile().intrinsics
    
#     depth_to_color_extrin =  profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.color))
#     color_to_depth_extrin =  profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( profile.get_stream(rs.stream.depth))

#     color_image = np.asanyarray(color_frame.get_data())
    
#     cv2.imshow("color" , color_image)
#     if cv2.waitKey(1) ==ord('q'):
#         break
    
#     time.sleep(3)
    
    
#     # print(intrinsics_color)
#     # print(intrinsics_depth)
#     print(depth_to_color_extrin)
#     print(color_to_depth_extrin)
    
    

# pipeline.stop()

# # 외부 파라미터 얻기
extrinsics = None
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        continue
    
    # 내부 파라미터 얻기
    intrinsics_color = color_frame.profile.as_video_stream_profile().intrinsics
    intrinsics_depth = depth_frame.profile.as_video_stream_profile().intrinsics


    # depth 프레임을 color 좌표계로 변환
    depth_frame = frames.get_depth_frame()
    depth_to_color_extrinsics = depth_frame.profile.get_extrinsics_to(color_frame.profile)
    depth_in_color_intrinsics = rs.video_stream_profile(intrinsics_color.width, intrinsics_color.height, rs.format.z16, 30).get_intrinsics()
    depth_frame = rs.apply_geometry_change(depth_frame, depth_in_color_intrinsics, depth_to_color_extrinsics)
    break

    # # depth 프레임에서 2D 좌표 얻기
    # depth_image = np.asanyarray(depth_frame.get_data())
    # x, y = 320, 240 # 예시 좌표
    # depth = depth_image[y, x] * depth_scale

    # # 2D 좌표를 3D 좌표로 변환
    # point = rs.rs2_deproject_pixel_to_point(intrinsics_depth, [x, y], depth)

    # # 외부 파라미터 계산
    # color_to_depth_extrinsics = depth_to_color_extrinsics.inverse()
    # point = np.array([point[0], point[1], point[2], 1.0]).reshape((4, 1))
    # extrinsics = np.zeros((4, 4))
    # extrinsics[3, 3] = 1.0
    # extrinsics[:3, :3] = np.array(color_to_depth_extrinsics.rotation).reshape((3, 3))
    # extrinsics[:3, 3] = np.array(color_to_depth_extrinsics.translation)
    # extrinsics = np.linalg.inv(extrinsics)
    # break

# 파이프라인 멈추기
pipeline.stop()
print(depth_to_color_extrinsics)
print(depth_in_color_intrinsics)