import pyrealsense2 as rs
import cv2
import numpy as np

pipeline = rs.pipeline()
config = rs.config()

# 640x480 해상도로 초기화
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline.start(config)

while True:
    # 현재 해상도 확인
   
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    color_image2 = cv2.resize(color_image , (640,480))
    
    cv2.imshow("color", color_image)
    cv2.imshow("color2" , color_image2)
    

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()