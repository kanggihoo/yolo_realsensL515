import numpy as np
import cv2
import pyrealsense2 as rs

# 체커보드 패턴의 가로 세로 점 개수
pattern_size = (6, 9)

# 체커보드 패턴의 3D 좌표 생성
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= 25  # 체커보드 셀의 크기: 25mm

# 체커보드 패턴 감지 및 3D 좌표 추출
def detect_chessboard_points(img, pattern_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        return corners

# 파이프라인 시작
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)

try:
    while True:
        # 프레임 가져오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 체커보드 패턴 감지 및 3D 좌표 추출
        corners = detect_chessboard_points(color_frame, pattern_size)
        if corners is not None:
            # 외부 파라미터 추정
            _, rvec, tvec = cv2.solvePnP(objp, corners, profile.get_intrinsics().as_matrix(), None)

            # rvec와 tvec를 출력
            print("rvec:", rvec)
            print("tvec:", tvec)

        # 화면에 보여주기
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("Color Image", color_image)
        cv2.waitKey(1)

finally:
    # 파이프라인 정리
    pipeline.stop()
    cv2.destroyAllWindows()