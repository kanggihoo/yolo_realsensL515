import os
import fileinput
import glob
path = r"C:\Users\11kkh\Desktop\roboflow\valid\labels"  # train, test , valid 변경

txt_file = glob.glob(path+"/*.txt")

for t in txt_file:
    print("name : " ,t)
    with fileinput.input(t , inplace=True) as f:
        for line in f:  
            stack = line.split()
            if stack[0] == '1':
                stack[0] ='0'
                line = str(' '.join(stack))
                print(line)