import cv2
import numpy as np
import math

image = cv2.imread("./img/tmp22.jpg")
print(f"image.shape : {image.shape}" )
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
coner_image = cv2.morphologyEx(gray , cv2.MORPH_CLOSE , kernel=kernel , iterations=10)
# 검출할 코너의 영역 설정(영역에 따라 코너 점이 달라질 수도 있어서 이부분에 대한 조치필요)
HEIGHT , WIDTH =  list(map(lambda x : int(x) , (image.shape[0]*0.2 , image.shape[1]*0.5)))
coner_image = gray[:HEIGHT , :WIDTH] 

# coner_image processing 
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# coner_image = cv2.morphologyEx(coner_image , cv2.MORPH_CLOSE , kernel=kernel , iterations=3)


## 여러개 코너 검출
# corners = cv2.goodFeaturesToTrack(coner_image, 10, 0.01, 5) # numpy type 반환
# corners = np.int32(corners)
# for i in corners:
#     corner = i.squeeze()
#     cv2.circle(image , corner , 3 , (0,0,255) ,-1)
#     print(corner)

# ## 1개의 코너 검출(FristPoint)
corners = cv2.goodFeaturesToTrack(coner_image, 1, qualityLevel=0.5 , minDistance=1) # numpy type 반환
corners = np.int32(corners)
CornerPoint = corners.squeeze()
corner_x , corner_y= CornerPoint

# # 값이 255 이면서 , X값이 WIDTH-1 일때 여러개의 y값들 중에서 가장 작을 때의 좌표값 구하기 # SecondPoint
# 이때 X값이 WIDTH-1일때만 비교하면 오차가 심해질 수 있음. 
y, x = np.where(coner_image ==255)

RightX_index = np.where(x == WIDTH-1)
RightY = min(y[RightX_index])
SecondPoint = (WIDTH-1 , RightY)
angle = math.atan2(-(SecondPoint[1] - CornerPoint[1]) , SecondPoint[0] - CornerPoint[0])*180/math.pi
print("angle : " , math.atan2(-(SecondPoint[1] - CornerPoint[1]) , SecondPoint[0] - CornerPoint[0])*180/math.pi)


if angle < 0 : # 코너점이 최상단점 (y값이 가장 작을때 )
    min_y = min(y)
    min_y_index = np.where(y==min_y)
    min_x = min(x[min_y_index])
    angle = math.atan2(-(SecondPoint[1] - min_y) , SecondPoint[0] - min_x)*180/math.pi
    TopPoint = (min_x , min_y)
    
    candidate = range(WIDTH-8 ,WIDTH)
    fianl_angle = 0 
    for i in candidate:
        
        RightX_index = np.where(x == i)
        RightY = min(y[RightX_index])
        SecondPoint = (i , RightY)
        angle = math.atan2(-(SecondPoint[1] - TopPoint[1]) , SecondPoint[0] - TopPoint[0])*180/math.pi
        if abs(fianl_angle) < abs(angle):
            fianl_angle = angle
        print("for 문 angle : " , math.atan2(-(SecondPoint[1] - TopPoint[1]) , SecondPoint[0] - TopPoint[0])*180/math.pi)
        
    print("new_angle : " , fianl_angle)
    print(f"CornerPoint : {CornerPoint}")
    print(f"top Point : {TopPoint} " )
    print(f"SecondPoint : {SecondPoint}")
    ## 끝점 구하기 
    RightSidePoint = image.shape[1]-1 , ((image.shape[1]-1) - TopPoint[0])*math.tan(math.radians(-fianl_angle)) + (TopPoint[1])
    print(f"RightSidePoint : " , RightSidePoint)
    print(math.atan2(-(RightSidePoint[1]-TopPoint[1]  ) , RightSidePoint[0] - TopPoint[0])*180/math.pi)
    RightSidePoint = np.int16(RightSidePoint)
    print(90-abs(fianl_angle))
    LeftSidePoint = 0 , (TopPoint[0])*math.tan(math.radians(90-abs(fianl_angle))) + TopPoint[1]
    print(f"LeftSidePoint : " , LeftSidePoint)
    LeftSidePoint = np.int16(LeftSidePoint)
    
    ## 길이 확인
    print("RightSide와 TopPoint 길이 : " , math.sqrt( ((RightSidePoint[0])-TopPoint[0])**2 + (RightSidePoint[1] - TopPoint[1])**2))
    print("LeftSide와 TopPoint  길이 : " , math.sqrt((LeftSidePoint[0]-TopPoint[0])**2 + (LeftSidePoint[1]- TopPoint[1])**2))
    
    
    ## 시각화
    # cv2.circle(image , (WIDTH-1 , RightY) , 3 , (255,0,0) , -1)
    cv2.circle(image , (WIDTH,HEIGHT) , 1 , (0,0,255) , -1)
    cv2.circle(image , TopPoint , 1 , (0,0,255) , -1)
    cv2.circle(image , RightSidePoint , 1 , (255,0,0) , -1)
    cv2.circle(image , LeftSidePoint , 1 , (255,0,0) , -1)
    
    cv2.line(image , TopPoint , RightSidePoint , (0,255,0) , 1)
    cv2.line(image, TopPoint , LeftSidePoint , (0,255,0) , 1)
else : # 코너점이 가장 좌측에 존재함. (x값이 가장 작을 때)
    min_x = min(x)
    min_x_index = np.where(x==min_x)
    min_y = min(y[min_x_index])
    angle = math.atan2(-(SecondPoint[1] - min_y) , SecondPoint[0] - min_x)*180/math.pi
    LeftPoint = (min_x , min_y)
    
    candidate = range(WIDTH-8 ,WIDTH)
    fianl_angle = 0 
    for i in candidate:
        
        RightX_index = np.where(x == i)
        RightY = min(y[RightX_index])
        SecondPoint = (i , RightY)
        angle = math.atan2(-(SecondPoint[1] - LeftPoint[1]) , SecondPoint[0] - LeftPoint[0])*180/math.pi
        if abs(fianl_angle) < abs(angle):
            fianl_angle = angle
        print("for 문 angle : " , math.atan2(-(SecondPoint[1] - LeftPoint[1]) , SecondPoint[0] - LeftPoint[0])*180/math.pi)
        
    print("new_angle : " , fianl_angle)
    print(f"CornerPoint : {CornerPoint}")
    print(f"Left Point : {LeftPoint} " )
    print(f"SecondPoint : {SecondPoint}")
    
    ## 끝점 구하기 
    
    RightSidePoint =  (LeftPoint[1])/math.sin(math.radians(angle)) , 0
    RightSidePoint2 =  (LeftPoint[1])/math.tan(math.radians(angle))+LeftPoint[0] , 0
    print(f"RightSidePoint : " , RightSidePoint)
    print(f"RightSidePoint2 : " , RightSidePoint2)
    print(math.atan2(-(RightSidePoint[1]-LeftPoint[1]  ) , RightSidePoint[0] - LeftPoint[0])*180/math.pi)
    RightSidePoint = np.int16(RightSidePoint)
    
    BotomPoint = (image.shape[0]-LeftPoint[1])*math.tan(math.radians(fianl_angle)) + LeftPoint[0],  image.shape[0] -1
    print(f"BotomPoint : " , BotomPoint)
    BotomPoint = np.int16(BotomPoint)
    
    ## 길이 확인
    print("RightSide와 LeftPoint 길이 : " , math.sqrt( ((RightSidePoint[0])-LeftPoint[0])**2 + (RightSidePoint[1] - LeftPoint[1])**2))
    print("LeftSide와 LeftPoint  길이 : " , math.sqrt((BotomPoint[0]-LeftPoint[0])**2 + (BotomPoint[1]- LeftPoint[1])**2))
    
    ## 시각화
    cv2.circle(image , (WIDTH-1 , RightY) , 3 , (255,0,0) , -1)
    cv2.circle(image , (WIDTH,HEIGHT) , 1 , (0,0,255) , -1)
    cv2.circle(image , LeftPoint , 1 , (0,0,255) , -1)
    cv2.circle(image , RightSidePoint , 1 , (255,0,0) , -1)
    cv2.circle(image , BotomPoint , 1 , (255,0,0) , -1)
    
    cv2.line(image , LeftPoint , RightSidePoint , (0,255,0) , 1)
    cv2.line(image, LeftPoint , BotomPoint , (0,255,0) , 1)
    
    
    


cv2.imshow("original" , image)
cv2.imshow("original2" , coner_image)
cv2.imshow("gray" , gray)


cv2.waitKey(0)
cv2.destroyAllWindows()