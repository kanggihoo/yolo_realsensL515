import os
import sys
from pathlib import Path
import glob 
import random
import shutil

directory_name = r"C:\Users\11kkh\Desktop\realsense_custom_data"
image_path = Path(directory_name) / "augmented_data"
label_path = Path(directory_name) / "augmented_label"


images = glob.glob(str(image_path)+"/*.jpg")
txt = glob.glob(str(label_path)+"/*.txt")
total_image = len(images)
random.seed(42)
images_index = [i for i in range(total_image)]
random.shuffle(images_index)

num_train = int(total_image * 0.7)
num_valid = int(total_image * 0.2)
num_test = total_image - (num_train + num_valid)

train_image_index = images_index[:num_train]
valid_image_index  = images_index[num_train:num_train+num_valid]
test_image_index = images_index[num_train+num_valid:]

def show_result():
    print("total_data 갯수 : " , total_image)
    print("num_train 갯수 : " , num_train)
    print("num_valid 갯수 : " , num_valid)
    print("num_test 갯수 : " , num_test)
    print()
    print("train_image  갯수 : " , len(train_image_index))
    print("valid_image  갯수 : " , len(valid_image_index))
    print("test_image  갯수 : " , len(test_image_index))

try:
    if total_image == len(train_image_index) + len(valid_image_index) + len(test_image_index):
        show_result()
    else:
        raise Exception
except Exception as e:
    print("data 분리가 잘 이루어 지지 않음")
    show_result()
    
# image 이름에 해당되는 text 데이터 이동 ##############################################################################################
train_image = []
train_txt = []
valid_image = []
valid_txt = []
test_image = []
test_txt = []
for idx , image in enumerate(images):
    if idx in train_image_index:
        train_image.append(image)
        train_txt.append(txt[idx])
    elif idx in valid_image_index:
        valid_image.append(image)
        valid_txt.append(txt[idx])
    else:
        test_image.append(image)
        test_txt.append(txt[idx])
    
def check(image , txt , data_inf = 'None'):
    for i in range(len(image)):
        
        assert Path(image[i]).stem == Path(txt[i]).stem , print("불일치 파일 존재!" ,image[i] , txt[i] )
    print(f"{data_inf} 문제없음")

check(train_image , train_txt , "train")            
check(valid_image , valid_txt , "valid")  
check(test_image , test_txt , "test")  


# 특정 디렉토리로 이동 #############
def move_file(files , copy_dir):
    copy_num = 0
    if not os.path.exists(copy_dir):
        os.mkdir(copy_dir)
    for file in files:
        file_name = Path(file).name
        copy_path = Path(copy_dir).joinpath(file_name)
        if not copy_path.exists:
            shutil.copy(file , str(copy_path))
            copy_num +=1
    print(f"총 {len(files)} 중 {copy_num}개 복사")
            

# move_file(test_image , r"C:\Users\11kkh\Desktop\data\data\test\images")
# move_file(test_txt , r"C:\Users\11kkh\Desktop\data\data\test\labels")

# move_file(train_image , r"C:\Users\11kkh\Desktop\data\data\train\images")
# move_file(train_txt , r"C:\Users\11kkh\Desktop\data\data\train\labels")

# move_file(valid_image , r"C:\Users\11kkh\Desktop\data\data\valid\images")
# move_file(valid_txt , r"C:\Users\11kkh\Desktop\data\data\valid\labels")
        