

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
sensor_dep = profile.get_device().first_depth_sensor()
depth_scale = sensor_dep.get_depth_scale()
print("Depth Scale is: " , depth_scale)
print("min_distance : ",sensor_dep.get_option(rs.option.min_distance))
# sensor_dep.set_option(rs.option.min_distance , 0)
print("set_min_distance : ",sensor_dep.get_option(rs.option.min_distance))
print("visual_preset : ",sensor_dep.get_option(rs.option.visual_preset))
# sensor_dep.set_option(rs.option.visual_preset , 3)
print("visual_preset : ",sensor_dep.get_option(rs.option.visual_preset))
print("depth_offset : ",sensor_dep.get_option(rs.option.depth_offset))
print("free_fall : ",sensor_dep.get_option(rs.option.freefall_detection_enabled))
print("sensor_mode : ",sensor_dep.get_option(rs.option.sensor_mode))
print("host_performance : ",sensor_dep.get_option(rs.option.host_performance))
print("max_distance : ",sensor_dep.get_option(rs.option.enable_max_usable_range))
print("noise_estimation: ",sensor_dep.get_option(rs.option.noise_estimation))
print("alternate_ir: ",sensor_dep.get_option(rs.option.alternate_ir))
print("digital_gain: ",sensor_dep.get_option(rs.option.digital_gain))
print("laser_power: ",sensor_dep.get_option(rs.option.laser_power))
print("confidence_threhold: ",sensor_dep.get_option(rs.option.confidence_threshold))
print("mid distance: ",sensor_dep.get_option(rs.option.min_distance))
print("receiver_gain: ",sensor_dep.get_option(rs.option.receiver_gain))
print("post_processing_shapening: ",sensor_dep.get_option(rs.option.post_processing_sharpening))
print("pre_processing_sharpening: ",sensor_dep.get_option(rs.option.pre_processing_sharpening))
print("noise_filtering: ",sensor_dep.get_option(rs.option.noise_filtering))





# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

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
epochs = 100
depth_image_sum= np.zeros(shape = (480 , 640) , dtype = np.int32)
try:
    for epoch in range(epochs):
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
        num_line = 11
        for i in range(num_line):
            cv2.rectangle(color_image , (0,239+(-(num_line//2)+i)*20) , (640-1,239+(-(num_line//2)+i)*20) , (255,0,0) , 1) # x축
            cv2.rectangle(color_image , (319+(-(num_line//2)+i)*20,0) , (319+(-(num_line//2)+i)*20 , 480-1) , (255,0,0) , 1) # y축
        
        cv2.circle(depth_colormap , (int(640/2)-1 , int(480/2)-1) , 3, (255,255,255) , -1)
        cv2.circle(color_image , (319 , 239) , 1, (255,255,255) , -1)
        cv2.circle(color_image , (0,0) , 3, (255,255,255) , -1)
        
        distance = aligned_depth_frame.get_distance(int(640/2)-1 , int(480/2)-1)
        ####
        # 1cm = 10.6 pixel
        # 10cm = 106 pixel
        # (429,267) , (429,161) => 10cm
        cm_per_pixel_ratio = 10/(267-161) # pixel to cm 관련 변수 #0.09433962264150944
        pixel_per_cm_ratio = 1/cm_per_pixel_ratio # 10.6
        
        
        center = (319,239)
        cv2.circle(color_image , (center[0]+106 , center[1]+106) , 1,(0,0,0),-1)

        
        
        depth_image_sum += depth_image
        
        
        # # print(distance)
        # cv2.imshow('depth_image', depth_colormap)
        # cv2.imshow('color_image' , color_image)
        # cv2.setMouseCallback('color_image' , mouse_click , color_image)
        # key = cv2.waitKey(1)
     
        
        # # # Press esc or 'q' to close the image window
        # if key & 0xFF == ord('q') or key == 27:
        #     cv2.destroyAllWindows()
        #     break
        print("epoch {} / {}".format(epoch , epochs))
        
finally:
    pipeline.stop()

depth_image_sum = depth_image_sum.astype(np.float64)

np.save(r'C:\Users\11kkh\Desktop\yolov5\depth_array' , depth_image_sum /epochs*depth_scale)
       
    


