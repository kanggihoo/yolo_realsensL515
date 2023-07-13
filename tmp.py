import cv2
import numpy as np
import math

def FindFourPoint(corner_image , HEIGHT , WIDTH):
# corner_image processing 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    corner_image = cv2.morphologyEx(corner_image , cv2.MORPH_CLOSE , kernel=kernel , iterations=3)
    y, x = np.where(corner_image ==255)

    ### 최상단 점 찾기 (p1)
    min_y  = min(y)
    min_y_index = np.where(y == min_y)
    min_x = min(x[min_y_index])
    cv2.circle(image , (min_x , min_y) ,2 , (255,0,0) , -1 )
    TopX , TopY = min_x , min_y
    p1 = (TopX , TopY)
    print("p1 최상단점 : ", p1 )

    ### x값이 가장 작은 경우(p2) y가 가장 작은 경우
    min_x  = min(x)
    min_x_index = np.where(x == min_x)
    min_y = min(y[min_x_index])
    cv2.circle(image , (min_x , min_y) ,2 , (255,0,0) , -1 )
    p2 = (min_x , min_y)
    print("p2 가장 좌측 점 : ", p2)


    ### 경계의 하단 부이면서 x가 가장 작은 경우 (p3)
    BotomY_index = np.where(y == HEIGHT-1)
    BotomX = min(x[BotomY_index])
    p3 = (BotomX , HEIGHT-1)
    cv2.circle(image , p3 ,2 , (255,0,0) , -1 )
    print("p3 경계 하단 점 : ", p3 )

    ### 경계의 우측 부이면서 y가 가장 작은 경우 (p4)
    RightX_index = np.where(x == WIDTH-1)
    RightY = min(y[RightX_index])
    p4 = (WIDTH-1 , RightY)
    cv2.circle(image , p4 ,2 , (255,0,0) , -1 )
    print("p4 경계 우측 점 : ", p4 )
    return p1,p2,p3,p4

def FindAngleAndLength(iamge, reference_point , HEIGHT , WIDTH):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    corner_image = cv2.morphologyEx(iamge , cv2.MORPH_CLOSE , kernel=kernel , iterations=3)
    y,x = np.where(corner_image ==255) 
    candidate = range(reference_point[0]+10 , WIDTH+20 ,2)   
    final_angle = 0 
    for i in candidate:
        RightX_index = np.where(x == i)
        RightY = min(y[RightX_index])
        RightPoint = (i , RightY)
        angle = math.atan2(-(RightPoint[1] - reference_point[1]) , RightPoint[0] - reference_point[0])*180/math.pi
        cv2.circle(image , RightPoint , 1 , (255,0,0) , -1)
        # print("angle : " , angle)
        final_angle += angle
    final_angle /= len(candidate)
    print("mean angle : " , final_angle)
    
    if final_angle <0:
        spare = reference_point[1]
        new_slice = image[spare:image.shape[0]-spare , spare: image.shape[1]-spare]
        reference_point = (lambda x : x-reference_point[1])(reference_point)
        
        RightSidePoint = new_slice.shape[1]-1 , ((new_slice.shape[1]-1) - reference_point[0])*math.tan(math.radians(-final_angle)) 
        RightSidePoint = np.int16(RightSidePoint)
        LeftSidePoint = 0 , (reference_point[0])*math.tan(math.radians(90-abs(final_angle))) 
        LeftSidePoint = np.int16(LeftSidePoint)
        LastPoint = math.sqrt( ((RightSidePoint[0])-reference_point[0])**2 + (RightSidePoint[1] - reference_point[1])**2)*math.cos(math.radians(-final_angle)) , new_slice.shape[0]-1
        LastPoint = np.int16(LastPoint)
        WidthLength =  math.sqrt( ((RightSidePoint[0])-reference_point[0])**2 + (RightSidePoint[1] - reference_point[1])**2)
        HeightLenght = math.sqrt((LeftSidePoint[0]-reference_point[0])**2 + (LeftSidePoint[1]- reference_point[1])**2)
        print("RightSide와 reference_point 길이 : " , WidthLength)
        print("LeftSide와 reference_point  길이 : " , HeightLenght)
    
        if WidthLength > HeightLenght:
            final_angle  = round(abs(final_angle)+90,1)
        else : 
            final_angle  = round(abs(final_angle),1)
            
            
        cv2.line(new_slice , reference_point , RightSidePoint , (0,255,0) ,2)
        cv2.line(new_slice, reference_point , LeftSidePoint , (0,255,0) , 2)
        cv2.line(new_slice, LeftSidePoint , LastPoint , (0,255,0) , 2)
    else :
        spare = reference_point[0]
        new_slice = image[spare:image.shape[0]-spare , spare: image.shape[1]-spare]
        reference_point = (lambda x : x-reference_point[0])(reference_point)
        
        
        RightSidePoint = (reference_point[1])/math.tan(math.radians(angle)) , 0
        RightSidePoint = np.int16(RightSidePoint)
        
        BotomPoint = (new_slice.shape[0]-1-reference_point[1])*math.tan(math.radians(final_angle)),  new_slice.shape[0] -1
        print(f"BotomPoint : " , BotomPoint)
        BotomPoint = np.int16(BotomPoint)
        
        
        LastPoint = math.sqrt( ((RightSidePoint[0])-reference_point[0])**2 + (RightSidePoint[1] - reference_point[1])**2)*math.cos(math.radians(angle))+BotomPoint[0] , (new_slice.shape[0]-1)-math.sqrt( ((RightSidePoint[0])-reference_point[0])**2 + (RightSidePoint[1] - reference_point[1])**2)*math.sin(math.radians(angle))
        LastPoint = np.int16(LastPoint)
        
        WidthLength =   math.sqrt( ((RightSidePoint[0])-reference_point[0])**2 + (RightSidePoint[1] - reference_point[1])**2)
        HeightLenght = math.sqrt((BotomPoint[0]-reference_point[0])**2 + (BotomPoint[1]- reference_point[1])**2)
        print("RightSide와 reference_point 길이 : " , WidthLength)
        print("LeftSide와 reference_point  길이 : " , HeightLenght)
        
        if WidthLength > HeightLenght:
            final_angle  = round(final_angle+90,1)
        else :
            final_angle  = round(final_angle,1)
        cv2.line(new_slice , reference_point , RightSidePoint , (0,255,0) ,2)
        cv2.line(new_slice, reference_point , BotomPoint , (0,255,0) , 2)
        cv2.line(new_slice, BotomPoint , LastPoint , (0,255,0) , 2)

    return final_angle , new_slice 
    
       

image = cv2.imread("./img/tmp11.jpg")
print(f"image.shape : {image.shape}" )
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

# 검출할 코너의 영역 설정(영역에 따라 코너 점이 달라질 수도 있어서 이부분에 대한 조치필요)
HEIGHT , WIDTH =  list(map(lambda x : int(x) , (image.shape[0]*0.5 , image.shape[1]*0.5)))
corner_image = gray[:HEIGHT , :WIDTH] 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
p1, p2 ,p3,p4 = FindFourPoint(corner_image , HEIGHT , WIDTH)

if p1[0] < p4[0] and p2[0] == p3[0]: # 좌측 상단에 최고점 존재(p1) => 구하고자 하는 각도 음수
    print("좌측 상단이 최고점")
    corner_image = gray[: , :] 
    final_angle , new_slice = FindAngleAndLength(corner_image, p1 , HEIGHT , WIDTH)
    print("final_angle : " , final_angle)
    
       

elif p1[1] == p4[1] and p2[0] < p3[0]: # 좌측 상단에 x값이 가장 작은 점이 있음(p2) => 구하고자 하는 각도 양수
    print("좌측 상단의 x가 가장 작음")
    ## p2를 기준으로 다시 영역을 설정한 뒤 각도 계산
    corner_image = gray[:, :] 
    corner_image = cv2.morphologyEx(corner_image , cv2.MORPH_CLOSE , kernel=kernel , iterations=3)
    
    final_angle, new_slice = FindAngleAndLength(corner_image, p2 , HEIGHT , WIDTH)
    print("final_angle : " , final_angle)
    
else : # 좌측을 더 넓게 탐색 
    print("예외 경우, 영역 재탐색 ")
    HEIGHT , WIDTH =  list(map(lambda x : int(x) , (image.shape[0]*0.7 , image.shape[1]*0.3)))
    corner_image = gray[:HEIGHT , :WIDTH]   
    p1,p2,p3,p4 = FindFourPoint(corner_image , HEIGHT , WIDTH)
    
    
    if p1[0] < p4[0] and p2[0] == p3[0]: # 좌측 상단에 최고점 존재(p1) => 구하고자 하는 각도 음수
        print("좌측 상단이 최고점")
        ## p1을 기준으로 다시 영역을 설정한 뒤 각도 계산
        corner_image = gray[: , :] 
        final_angle , new_slice = FindAngleAndLength(corner_image, p1 , HEIGHT , WIDTH)
        print("final_angle : " , final_angle)

    elif p1[1] == p4[1] and p2[0] < p3[0]: # 좌측 상단에 x값이 가장 작은 점이 있음(p2) => 구하고자 하는 각도 양수
        print("좌측 상단의 x가 가장 작음")
        ## p2를 기준으로 다시 영역을 설정한 뒤 각도 계산
        corner_image = gray[:, :] 
        corner_image = cv2.morphologyEx(corner_image , cv2.MORPH_CLOSE , kernel=kernel , iterations=3)
        
        final_angle, new_slice = FindAngleAndLength(corner_image, p2 , HEIGHT , WIDTH)
        print("final_angle : " , final_angle)


cv2.imshow('Corners', image)
cv2.imshow('Corners2', corner_image)
cv2.imshow("new" , new_slice)
cv2.waitKey()
cv2.destroyAllWindows()



