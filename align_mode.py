import pyrealsense2 as rs
import cv2
import keyboard
while 1:
    key = keyboard.read_key()
    if key == 'p':
        print("p 방향키 누름")
        
        # 카메라 초기화
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 0, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 0, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

        # 프레임 캡처 및 rs-align 모드 활성화
        align = rs.align(rs.stream.depth)

        while True:
            # 프레임 가져오기q
            frames = pipeline.wait_for_frames()
            
            # 색상 프레임과 깊이 프레임을 얻음
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # 2D 픽셀 좌표를 3D 좌표로 변환
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
            
            # 픽셀 좌표 설정 (예: (320, 240))
            pixel_x = 320
            pixel_y = 240
            
            # 2D 픽셀 좌표를 3D 좌표로 변환
            depth = depth_frame.get_distance(pixel_x, pixel_y)
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [pixel_x, pixel_y], depth)
            color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
            
            # 변환된 3D 좌표 출력
            print("3D 좌표 (x, y, z):", color_point)
            print("3D depth " , depth_point)
            
            # 종료 조건 설정 (예: 'q' 키 입력 시 종료)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        # 프로그램 종료
        pipeline.stop()
