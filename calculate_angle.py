import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

img = cv2.imread(r"C:\Users\11kkh\Desktop\yolov5\img\test.png")
noise_rect =  cv2.imread(r"C:\Users\11kkh\Desktop\yolov5\img\noise_rect.png")
img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
noise_rect_gray = cv2.cvtColor(noise_rect , cv2.COLOR_BGR2GRAY)


# 배경이 검정색, 검출하고자 하는 이미지의 픽셀이 흰색이면 더 잘 검출된다. 
################################################################################# 일반적인 사각형에 대한 4개의 꼭짓점 검출
start = time.time()
contours , hierarchy = cv2.findContours(img_gray, mode = cv2.RETR_LIST , method = cv2.CHAIN_APPROX_SIMPLE ) 

for c in contours:
    c = np.array(c).squeeze()
    print(c.shape)
    x = c[:,0]
    y = c[:,1]
    x_min, y_min= min(x) , min(y)
    x_max, y_max= max(x) , max(y)
    x_min_indexes = np.where(x == x_min)[0]
    x_max_indexes = np.where(x == x_max)[0]
    y_min_indexes = np.where(y == y_min)[0]
    y_max_indexes = np.where(y == y_max)[0]
    
    #p1 : y가 가장 작을때의 x값 , 그때의 x값이 여러개 존재하는 경우에 대해서는 가장 작은 x값 
    #p2 : x가 가장 클때의 y값 , 그때의 y값이 여러개 존재하는 경우에 대해서는 가장 작은 y값
    #p3 : y가 가장 클때의 x값 , 그때의 x값이 여러개 존재하는 경우에 대해서는 가장 큰 x값
    #p4 : x가 가장 작을때 y값 , 그때의 y값이 여러개 존재하는 경우에 대해서는 가장 큰 y값
    
    #p1 구하기
    p1_x_condidate , p1_y_condidate = x[y_min_indexes] , y[y_min_indexes]
    p1_x , p1_x_index = min(p1_x_condidate) , np.argmin(p1_x_condidate)
    p1_y = p1_y_condidate[p1_x_index]
    cv2.circle(img , (p1_x,p1_y) ,2,(255,0,0) , 2 )
    cv2.putText(img , "p1" , (p1_x , p1_y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(255,0,0))
    
    #p2 구하기 
    p2_y_condidate , p2_x_condidate = y[x_max_indexes] , x[x_max_indexes]
    p2_y , p2_y_index = min(p2_y_condidate), np.argmin(p2_y_condidate)
    p2_x = p2_x_condidate[p2_y_index]
    cv2.circle(img , (p2_x,p2_y) ,2,(255,0,0) , 2 )
    cv2.putText(img , "p2" , (p2_x , p2_y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(255,0,0))
    
    #p3 구하기
    p3_x_condidate , p3_y_condidate = x[y_max_indexes] , y[y_max_indexes]
    p3_x , p3_x_index = max(p3_x_condidate) , np.argmax(p3_x_condidate)
    p3_y = p3_y_condidate[p3_x_index]
    cv2.circle(img , (p3_x,p3_y) ,2,(255,0,0) , 2 )
    cv2.putText(img , "p3" , (p3_x , p3_y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(255,0,0))
    
    #p4 구하기 
    p4_y_condidate , p4_x_condidate = y[x_min_indexes] , x[x_min_indexes]
    p4_y , p4_y_index = max(p4_y_condidate), np.argmax(p4_y_condidate)
    p4_x = p4_x_condidate[p4_y_index]
    cv2.circle(img , (p4_x,p4_y) ,2,(255,0,0) , 2 )
    cv2.putText(img , "p4" , (p4_x , p4_y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(255,0,0))
    
################################################################################# 노이즈가 있는 사각형에 대한 4개의 꼭짓점 검출    
contours , hierarchy = cv2.findContours(noise_rect_gray, mode = cv2.RETR_LIST , method = cv2.CHAIN_APPROX_SIMPLE ) 

noise_compensation = 3 # 2pixel
for c in contours:
    c = np.array(c).squeeze()
    print(c.shape)
    x = c[:,0]
    y = c[:,1]
    x_min, y_min= min(x) , min(y)
    x_max, y_max= max(x) , max(y)
    x_min_indexes = np.where(x <= x_min+noise_compensation)[0]
    x_max_indexes = np.where(x >= x_max-noise_compensation)[0]
    y_min_indexes = np.where(y <= y_min+noise_compensation)[0]
    y_max_indexes = np.where(y >= y_max-noise_compensation)[0]
    
    #y,x의 가장 작을 때보다는 노이즈 고려해서 
    #p1 : y가 가장 작을때의 x값 , 그때의 x값이 여러개 존재하는 경우에 대해서는 가장 작은 x값 
    #p2 : x가 가장 클때의 y값 , 그때의 y값이 여러개 존재하는 경우에 대해서는 가장 작은 y값
    #p3 : y가 가장 클때의 x값 , 그때의 x값이 여러개 존재하는 경우에 대해서는 가장 큰 x값
    #p4 : x가 가장 작을때 y값 , 그때의 y값이 여러개 존재하는 경우에 대해서는 가장 큰 y값
    
    #p1 구하기
    p1_x_condidate , p1_y_condidate = x[y_min_indexes] , y[y_min_indexes]
    p1_x , p1_x_index = min(p1_x_condidate) , np.argmin(p1_x_condidate)
    p1_y = p1_y_condidate[p1_x_index]
    cv2.circle(noise_rect , (p1_x,p1_y) ,2,(255,0,0) , 2 )
    cv2.putText(noise_rect , "p1" , (p1_x , p1_y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(255,0,0))
    
    #p2 구하기 
    p2_y_condidate , p2_x_condidate = y[x_max_indexes] , x[x_max_indexes]
    p2_y , p2_y_index = min(p2_y_condidate), np.argmin(p2_y_condidate)
    p2_x = p2_x_condidate[p2_y_index]
    cv2.circle(noise_rect , (p2_x,p2_y) ,2,(255,0,0) , 2 )
    cv2.putText(noise_rect , "p2" , (p2_x , p2_y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(255,0,0))
    
    #p3 구하기
    p3_x_condidate , p3_y_condidate = x[y_max_indexes] , y[y_max_indexes]
    p3_x , p3_x_index = max(p3_x_condidate) , np.argmax(p3_x_condidate)
    p3_y = p3_y_condidate[p3_x_index]
    cv2.circle(noise_rect , (p3_x,p3_y) ,2,(255,0,0) , 2 )
    cv2.putText(noise_rect , "p3" , (p3_x , p3_y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(255,0,0))
    
    #p4 구하기 
    p4_y_condidate , p4_x_condidate = y[x_min_indexes] , x[x_min_indexes]
    p4_y , p4_y_index = max(p4_y_condidate), np.argmax(p4_y_condidate)
    p4_x = p4_x_condidate[p4_y_index]
    cv2.circle(noise_rect , (p4_x,p4_y) ,2,(255,0,0) , 2 )
    cv2.putText(noise_rect , "p4" , (p4_x , p4_y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(255,0,0))


    
    
    
end = time.time()-start
cv2.imshow("img" , img)
cv2.imshow("noise_img" , noise_rect)
print(end)
cv2.waitKey()
cv2.destroyAllWindows()

