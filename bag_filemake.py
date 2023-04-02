import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, 'test.bag')

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 15)

profile = pipeline.start(config)

filt = rs.save_single_frameset()

for x in range(100):
    pipeline.wait_for_frames()

frame = pipeline.wait_for_frames()
filt.process(frame)