from pathlib import Path
import os
import glob

#image file directory
image_directory  = r"C:\Users\11kkh\Desktop\realsense_custom_data\augmented_data" 
label_directory = r"C:\Users\11kkh\Desktop\realsense_custom_data\augmented_label"

#all image_name in image directory
images = glob.glob(r"C:\Users\11kkh\Desktop\realsense_custom_data\augmented_data\*.jpg" )
labels = glob.glob(r"C:\Users\11kkh\Desktop\realsense_custom_data\augmented_label\*.txt" )
print("number of the total images before remove :" , len(images))
print("number of the total labels before remove :" , len(labels))

for image in images:
    image_name = Path(image).stem
    
    compare = Path(label_directory).joinpath(str(image_name)+".txt")
    if compare.exists():
        continue
    else:
        print(image)
        os.remove(str(image))
    
print("image 파일 삭제 완료")
images = glob.glob(r"C:\Users\11kkh\Desktop\realsense_custom_data\augmented_data\*.jpg" )
labels = glob.glob(r"C:\Users\11kkh\Desktop\realsense_custom_data\augmented_label\*.txt" )
print("number of the total images after yot remove images:" , len(images))
print("number of the total labels after you remove images :" , len(labels))