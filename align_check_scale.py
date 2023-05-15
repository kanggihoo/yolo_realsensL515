

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




# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away


# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

sensor_dep = profile.get_device().first_depth_sensor()
depth_scale = sensor_dep.get_depth_scale()
sensor_dep.set_option(rs.option.min_distance , 490)
sensor_dep.set_option(rs.option.visual_preset , 0.0)  
sensor_dep.set_option(rs.option.error_polling_enabled , 1)  
sensor_dep.set_option(rs.option.enable_max_usable_range , 0.0)  
sensor_dep.set_option(rs.option.digital_gain , 1)  
sensor_dep.set_option(rs.option.laser_power , 89)  
sensor_dep.set_option(rs.option.confidence_threshold , 2)  
sensor_dep.set_option(rs.option.min_distance , 490)  
sensor_dep.set_option(rs.option.post_processing_sharpening , 1)  
sensor_dep.set_option(rs.option.pre_processing_sharpening , 2)  
sensor_dep.set_option(rs.option.noise_filtering , 3)  
sensor_dep.set_option(rs.option.invalidation_bypass , 1) 

########################################################

########
def mouse_click(event , x,y,flags , param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy_depth = aligned_depth_frame.get_distance(x, y)
        xy_3Dpoint = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], xy_depth)
        print("mouse_x_y : ", (x,y) ,"distance : ", xy_depth , "x_y_3D : ",xy_3Dpoint)
        center_depth = aligned_depth_frame.get_distance(319, 239)
        center_3Dpoint = rs.rs2_deproject_pixel_to_point(depth_intrin, [319, 239], center_depth)
        print("cneter : (319,239)", "center_distance : ", center_depth , "center_3D : ",center_3Dpoint)
        
        dif_x = abs(center_3Dpoint[0] - xy_3Dpoint[0])
        dif_y = abs(center_3Dpoint[1] - xy_3Dpoint[1])
        dif_z = abs(center_3Dpoint[2] - xy_3Dpoint[2])
        print(f"dif_x : {dif_x} , dif_y : {dif_y} , dif_z : {dif_z}")
        cv2.circle(param , (x,y) , 3 , (255,255,255),2)
    
        

# Streaming loop
# epochs = 100

try:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
        
finally:
    pipeline.stop()
    depth = np.round(np.asanyarray(aligned_depth_frame.get_data()) * depth_scale,5)
    np.save(r'C:\Users\11kkh\Desktop\yolov5\depth_array' , depth )
    print("저장 완료!")
       
    


