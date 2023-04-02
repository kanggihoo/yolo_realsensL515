import pyrealsense2 as rs

# 카메라 연결
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()

# 사용 가능한 옵션 리스트 출력
for option in depth_sensor.get_supported_options():
    print(option)