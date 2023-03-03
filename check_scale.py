## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


def mouse_click(event , x,y,flags , param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth = aligned_depth_frame.get_distance(x, y)
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth)
        print("distance : ", depth , "depth_point : ",depth_point)
        center = aligned_depth_frame.get_distance(319, 239)
        depth_point_center = rs.rs2_deproject_pixel_to_point(depth_intrin, [319, 239], center)
        print("중심 : " , center , depth_point_center)
        cv2.circle(param , (x,y) , 3 , (255,255,255),2)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # # frames.get_depth_frame() is a 640x360 depth image

        # # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())


        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        

        # # Render images:
        # #   depth align to color on left
        # #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # images = np.hstack((bg_removed, depth_colormap))
        
        cv2.circle(depth_colormap , (int(640/2)-1 , int(480/2)-1) , 3, (255,255,255) , -1)
        cv2.circle(color_image , (319 , 239) , 1, (255,255,255) , -1)
        cv2.circle(color_image , (0,0) , 3, (255,255,255) , -1)
        cv2.rectangle(color_image , (0,0) , (319,239) , (255,0,0) , 1)
        cv2.rectangle(color_image , (319,239) , (639,479) , (255,0,0) , 1)
        cv2.line(color_image , (0 , 239) ,   (319+60 , 239) , (255,255,255),1)
        cv2.line(color_image , (319 , 0) ,   (319 , 319+30) , (255,255,255),1)
        
        distance = aligned_depth_frame.get_distance(int(640/2)-1 , int(480/2)-1)
        ####
        # 1cm = 10.6 pixel
        # 10cm = 106 pixel
        # (429,267) , (429,161) => 10cm
        cm_per_pixel_ratio = 10/(267-161) # pixel to cm 관련 변수 #0.09433962264150944
        pixel_per_cm_ratio = 1/cm_per_pixel_ratio # 10.6
        
        
        center = (319,239)
        cv2.circle(color_image , (center[0]+106 , center[1]+106) , 1,(0,0,0),-1)
        
        
        # print(distance)
        cv2.imshow('depth_image', depth_colormap)
        cv2.imshow('color_image' , color_image)
        cv2.setMouseCallback('color_image' , mouse_click , color_image)
        key = cv2.waitKey(1)
        # # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
    