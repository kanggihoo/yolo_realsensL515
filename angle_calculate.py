

# import cv2
# import numpy as np
# import math

# # 이미지 불러오기
# image = cv2.imread('tmp2.png', 0)

# # 이미지 이진화
# _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# # 외곽선 검출
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 가장 큰 외곽선 추출
# largest_contour = max(contours, key=cv2.contourArea)

# # 외곽선 근사화
# epsilon = 0.02 * cv2.arcLength(largest_contour, True)
# approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# # 사각형 그리기
# first_index = True
# pre = None
# for i in approx:
#     i = i[0]
#     if first_index:
#         first_index = False
#         pre = (i[0] , i[1])
#         continue
#     cur = (i[0] , i[1])
#     print(f"pre :{pre} , cur = {cur}")
#     print(f"angle = {math.atan2(pre[0] - cur[0] , pre[1] - cur[1])*180/math.pi}")
#     pre = cur
    
    
        
#     i = i[0]
    
# # cv2.drawContours(image, [approx], 0, (255, 0, 0), 2)
# iamge = cv2.cvtColor(image , cv2.COLOR_GRAY2BGR)
# for i in approx:
#     cv2.circle(image , (i[0][0] , i[0][1]), 5 , (0,0,0),-1 )
# # 꼭짓점 좌표 및 회전 각도 계산
# # rect = cv2.minAreaRect(approx)

# # image = cv2.cvtColor(image , cv2.COLOR_GRAY2BGR)

# # box = cv2.boxPoints(rect)

# # box = np.int0(box)
# # cv2.drawContours(image , [box] , -1 , (255,0,0) , 3 )
# # angle = rect[-1]
# # tmp = np.zeros((270,270) , dtype = np.uint8)
# # tmp = cv2.cvtColor(tmp , cv2.COLOR_GRAY2BGR)
# # print(tmp.shape)
# # print("꼭짓점 좌표:")
# # for point in box:
# #     print(point)
# #     cv2.circle(image , (point[0] , point[1]) , 5 , (0,0,255) , -1 )
# #     cv2.circle(tmp , (point[0] , point[1]) , 5 , (0,0,255) , -1 )

# # print("회전 각도:", angle)

# cv2.imshow("sdsds" , image)
# # cv2.imshow("sdsdssds" , tmp)
# # 결과 출력
# cv2.waitKey()
# cv2.destroyAllWindows()

## 나머지 꼭짓점 좌표를 찾을 수 있다면 각도 조건 90도를 이용하여 임의의 직선을 그어서 확인?

# 1. 4개의 꼭짓점을 찾은경우 break => 4개꼭지점이 이루는 각도가 90도 사이인경우 (areaminbox) => Canny ende를 통해 외각선 또는 contoru찾은뒤 각도의 변화량이
# 명확하게 4개 존재하는 경우
# 2. 4개의 명확한 꼭짓점을 찾지 못한경우 
 # 2.1 => 시계방향회전 or 반시계 회전 각각의 경우에 따른 경우 확인
 # 1개의 꼭짓점을 명확하게 알 수 있을 것이고 나머지 2개 꼭짓점은 x축, y축과 만나면서 tmp2이미지 같은경우 
 
 
 


# image = cv2.imread('img/tmp4.jpg', 0)

# # 이미지 이진화
# _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# # 외곽선 검출
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 가장 큰 외곽선 추출
# largest_contour = max(contours, key=cv2.contourArea)

# image = cv2.cvtColor(image , cv2.COLOR_GRAY2RGB)
# cv2.drawContours(image, [largest_contour] , -1 , (255,0,0) , 2 )
# c = np.array(largest_contour).squeeze()
# x = c[:,0]
# y = c[:,1]
# x_min, y_min= min(x) , min(y)
# x_max, y_max= max(x) , max(y)
# x_min_indexes = np.where(x == x_min)[0]
# x_max_indexes = np.where(x == x_max)[0]
# y_min_indexes = np.where(y == y_min)[0]
# y_max_indexes = np.where(y == y_max)[0]

## 방법 1
#p1 : y가 가장 작을때의 x값 , 그때의 x값이 여러개 존재하는 경우에 대해서는 가장 작은 x값 
#p2 : x가 가장 클때의 y값 , 그때의 y값이 여러개 존재하는 경우에 대해서는 가장 작은 y값
#p3 : y가 가장 클때의 x값 , 그때의 x값이 여러개 존재하는 경우에 대해서는 가장 큰 x값
#p4 : x가 가장 작을때 y값 , 그때의 y값이 여러개 존재하는 경우에 대해서는 가장 큰 y값

# p1 구하기
# p1_x_condidate , p1_y_condidate = x[y_min_indexes] , y[y_min_indexes]
# p1_x , p1_x_index = min(p1_x_condidate) , np.argmin(p1_x_condidate)
# p1_y = p1_y_condidate[p1_x_index]
# cv2.circle(image , (p1_x,p1_y) ,2,(0,0,255) , 2 )
# cv2.putText(image , "p1" , (p1_x , p1_y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(0,0,255))

# #p2 구하기 
# p2_y_condidate , p2_x_condidate = y[x_max_indexes] , x[x_max_indexes]
# p2_y , p2_y_index = min(p2_y_condidate), np.argmin(p2_y_condidate)
# p2_x = p2_x_condidate[p2_y_index]
# cv2.circle(image , (p2_x,p2_y) ,2,(0,0,255) , 2 )
# cv2.putText(image , "p2" , (p2_x , p2_y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(0,0,255))

# #p3 구하기
# p3_x_condidate , p3_y_condidate = x[y_max_indexes] , y[y_max_indexes]
# p3_x , p3_x_index = max(p3_x_condidate) , np.argmax(p3_x_condidate)
# p3_y = p3_y_condidate[p3_x_index]
# cv2.circle(image , (p3_x,p3_y) ,2,(0,0,255) , 2 )
# cv2.putText(image , "p3" , (p3_x , p3_y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(0,0,255))

# #p4 구하기 
# p4_y_condidate , p4_x_condidate = y[x_min_indexes] , x[x_min_indexes]
# p4_y , p4_y_index = max(p4_y_condidate), np.argmax(p4_y_condidate)
# p4_x = p4_x_condidate[p4_y_index]
# cv2.circle(image , (p4_x,p4_y) ,2,(0,0,255) , 2 )
# cv2.putText(image , "p4" , (p4_x , p4_y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7,(0,0,255))

# print(f"p1 , p2 , p3 , p4 {p1_x,p1_y} ,{p2_x,p2_y} , {p3_x,p3_y} , {p4_x,p4_y}")
# rect_point = [[p1_x,p1_y],[p2_x,p2_y],[p3_x,p3_y],[p4_x,p4_y]]
# pre = None
# for idx , i in enumerate(rect_point):
#     if idx == 0:
#         pre = rect_point[idx]
#         continue
#     cur = i
#     print(f"cur = {i} , pre = {pre}")
    
#     # 각도계산
#     print(f"angle = {round(math.atan2(  -(cur[1] - pre[1]) , cur[0] - pre[0])*180/math.pi)}") # atan2(y,x)
#     # print(f"angle = {round(math.atan( cur[1] - pre[1] / cur[0] - pre[0] )*180/math.pi)}")
#     pre = cur
    
#     if idx == len(rect_point) -1:
#         cur = rect_point[0]
#         print(f"cur = {i} , pre = {pre}")
#         print(f"angle = {round(math.atan2(  -(cur[1] - pre[1]) , cur[0] - pre[0])*180/math.pi)}") # atan2(y,x)
        
    
    
    


# # ## 방법2 근사화
import cv2
import math
import numpy as np

def FindLargeContour_BoxPoint(image):
    '''
    image : binary_image
    '''
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    # image = cv2.cvtColor(image , cv2.COLOR_GRAY2RGB)
    rect = cv2.minAreaRect(largest_contour)
    box_point = cv2.boxPoints(rect) 
    return  largest_contour , box_point 

# IOU 구하기
def FindIOU(binary , largest_contour , box_point):
    intersection = np.zeros_like(binary)
    cv2.drawContours(intersection, [largest_contour], -1, 255, thickness=cv2.FILLED)
    largest_contour_area = np.sum(intersection)/255

    intersection[:,:]=0
    cv2.drawContours(intersection, [box_point], -1, 255, thickness=cv2.FILLED)
    box_contour_area=np.sum(intersection)/255

    cv2.drawContours(intersection, [largest_contour], -1, 255, thickness=cv2.FILLED)
    union_area = np.sum(intersection)/255
    intersection_area = largest_contour_area +box_contour_area - np.sum(intersection)/255
    IOU = round(intersection_area/union_area , 2)
    print("합집합 넓이 : ",union_area)
    print("largest_contour 면적: " , largest_contour_area)
    print("box contour 면적 : " , box_contour_area)
    print("교집합 넓이:", intersection_area )
    print("IOU" , IOU)
    
    return IOU
def FindAngle(box_point):
    first_Point , second_Point , third_Point , last_Point = box_point
    angle = 0
    add_90angle = False
    if first_Point[1] > second_Point[1]: # 회전한 경우 
        if  ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) < ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):# height > width 
            angle = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi+90
            add_90angle = True
        elif ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) > ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):
            angle = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi
        else : # 길이 같은 경우 
            angle = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi # 회전했는데 가로, 세로길이 같은경우에는 예각, 둔각 회전 가능하지만 예각으로 회전
    elif first_Point[1] == second_Point[1]: ## 회전하지 않은 경우
        if  ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) < ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2): # height > width 
            angle = 90
            add_90angle = True
        elif ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) > ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):
            angle = 0
        else : # height == width 
            angle = 0
    return round(angle,1) , add_90angle

img_origin = cv2.imread('img/tmp16.jpg',0)
image = cv2.GaussianBlur(img_origin, (3, 3),sigmaX=0,sigmaY=0) # 그림 그리기 용 iamge)3채널? 아마도?
_, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
largest_contour , box_point = FindLargeContour_BoxPoint(binary)
box_int = np.intp(box_point)
first_Point , second_Point , third_Point , last_Point = box_int
IOU = FindIOU(binary , largest_contour , box_int)
angle , add_90angle = FindAngle(box_int)
image = cv2.cvtColor(image , cv2.COLOR_GRAY2BGR)
final_angle = None

## 각도 구하기
if IOU < 0.8: 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (5,5))
    binary2 = cv2.erode(binary , kernel , iterations=10)
    largest_contour2 , box_point2 = FindLargeContour_BoxPoint(binary2)
    box_int2= np.intp(box_point2)
    first_Point2 , second_Point2 , third_Point2 , last_Point2 = box_int2
    ## 각도 구하기 (만약 IOU가 0.8 미만인데 각도가 0도가 아닌경우 는 첫번째 , 두번째 각도의 평균으로 계산)
    angle2 , add_90angle2 = FindAngle(box_int2)
    print("angle_flag2 : " , add_90angle2)
    cv2.drawContours(image , [box_int2] , -1 , (0,0,255) , 2  )
    if angle < 5:
        final_angle = angle2
        print(f"angle : {angle} , angle2 : {angle2}")
        print("final_angel = angle2" , final_angle)
        
        ## 이렇게 해도 angle 2가 0나오는 경우 있음 완전 노이즈 있는 경우 
    else:
        final_angle = round((angle+angle2)/2 , 1)
        print(f"angle : {angle} , angle2 : {angle2}")
        print("final_angel = angle+angle2" , final_angle)
                
else :
    final_angle = angle
    
    
print("angle_flag : " , add_90angle)
print("final_angle : " , final_angle)
cv2.drawContours(image , [box_int] , -1 , (255,0,0) , 2  )

    

cv2.imshow("sdsd" , image)
cv2.imshow("origin" , img_origin)
cv2.imshow("bianry" , binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    
    
    
    
    
## 모든 외곽선 검사    
# stack = []
# count = 0
# stop = None
# Mode = 0
# for idx , i  in enumerate(largest_contour):
#     x , y = i[0]
#     pre_x,pre_y = x,y
    
    
#     if len(stack):
#         pre_x , pre_y = stack[-1]
#     print("x,y")
#     print(math.atan2(-(y-pre_y) , x-pre_x)*180/math.pi)
#     if Mode == 0 and ~(y-pre_y >=0 and x-pre_x <=0):
#         count += 1
#     if Mode == 1 and ~(y-pre_y >=0 and x-pre_x <=0):
#         count +=1 
    
    
#     if count > 2:
#         print(stack[-1])
#         stop = stack[-1]
#         cv2.circle(image , stop , 3,  (255,0,0) , -1)
#         Mode +=1 
#         count = 0 
#     if Mode == 1:
#         break
#     if len(stack) >5:
#         stack.pop(0) 
#     stack.append((x,y))

## 직선의 방정식 이용
# first_Point_ischange = False
# second_Point_ischange = False
# third_Point_ischange = False
# # first_Point가 이미지 크기의 범위를 벗어난 경우 
# if first_Point[0] < 0: 
#     first_Point_ischange = True
#     print("first_point X over range")
# if second_Point[1] <0:
#     second_Point_ischange = True
#     print("second_point Y over range")
# if third_Point[0] > image.shape[0]:
#     print("third_point X over range")
#     third_Point_ischange = True
# print("before point : " , first_Point , second_Point , third_Point)
# # 직선의 방정식(first , second)
# m1 = (second_Point[0] - first_Point[0]) / (second_Point[1] - first_Point[1])
# b1 = -m1*first_Point[0] + first_Point[1]

# m2 = (third_Point[0] - second_Point[0]) / (third_Point[1] - second_Point[1])
# b2 = -m2*second_Point[0] + second_Point[1]

# if first_Point_ischange:
#     first_Point[0] = 0
#     first_Point[1] = m1*first_Point[0]+b1
# if second_Point_ischange:
#     second_Point[1] = 0
#     second_Point[0] = m2*second_Point[1]+b2
# if third_Point_ischange:
#     third_Point[0] = image.shape[0]
#     third_Point[1] = m1*third_Point[0]+b1
# print("after point : " , first_Point , second_Point , third_Point)

# 시와 토마시 코너 검출 (corner_goodFeature.py)
# import cv2
# import numpy as np

# img = cv2.imread('img/tmp9.jpg')
# print(img.shape)
# range_slice = 70-1
# im0 = img.copy()[:range_slice+1,:range_slice+1]

# gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)

# # # 시-토마스의 코너 검출 메서드
# corners = cv2.goodFeaturesToTrack(gray, 1, 0.01, 10)
# # 실수 좌표를 정수 좌표로 변환
# corners = np.int32(corners)
# corner_x , corner_y= corners[0][0] 
# print("corners:",corners)
# for i in corners:
#     cv2.circle(im0 , i[0] , 4,(0,0,255) , -1)
# # 코너 픽셀 세밀화 함수
# # criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS , 30 ,0.001)
# # cv2.cornerSubPix(gray , corners, (5,5) , (-1,-1) , criteria)
# # print("코너점" , corners)
# # # 좌표에 동그라미 표시
# # for i in corners:
# #     cv2.circle(img, np.int32(i[0]), 2, (0,0,255), -1)


# ## 2차원 배열 값이 255이면서 최상단 점 찾기
# '''
# 1. 2차원 배열 중에서 값이 255인 행, 열 번호 찾기
# 2. 행 번호들 중에서 가장 작은 행 번호 찾고
# 3. 2번에서 찾은 행의 인덱스에 대응하는 열 번호 조합 찾기
# 4. 3번의 열 번호 조합에서 가장 작은 열 값을 찾기 
# '''
# y , x = np.where(gray ==255)

# ## 최상단 점 찾기 
# min_y  = min(y)
# min_y_index = np.where(y == min_y)
# min_x = min(x[min_y_index])
# cv2.circle(im0 , (min_x , min_y) ,2 , (255,0,0) , -1 )
# cv2.circle(img , (min_x , min_y) ,2 , (255,0,0) , -1 )
# TopX , TopY = min_x , min_y
# print("최상단점 : ", TopX , TopY )

# ## col:49이면서 행이 가장 작은 값
# x_49_index = np.where(x == range_slice)
# RightSideY = min(y[x_49_index])
# cv2.circle(im0 , (range_slice, RightSideY) , 2,(0,255,0) , -1)
# cv2.circle(img , (range_slice, RightSideY) , 2,(0,255,0) , -1)
# print(f"x={range_slice} 일때 : {range_slice}, {RightSideY}" )

# ## row:49이면서 열이 가장 작은 값
# y_49_index = np.where(y == range_slice)
# BottomSideX = min(x[y_49_index])
# cv2.circle(im0 , (BottomSideX, range_slice) , 2,(0,255,0) , -1)
# cv2.circle(img , (BottomSideX, range_slice) , 2,(0,255,0) , -1)
# print(f"y={range_slice} 일때 : ({BottomSideX},{range_slice}) " )


# if corner_y <= RightSideY:
#     first_Point = (BottomSideX , range_slice)
#     second_Point = (corner_x , corner_y)
# elif corner_y > RightSideY:
#     first_Point = (corner_x , corner_y)
#     second_Point = (range_slice , RightSideY)

## angle 구하기 
# if first_Point[1] > second_Point[1] : # 회전이 발생
#     if img.shape[0] > 

#         angle_point = math.atan2(-(second_Point[1]-first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi
    

# elif first_Point[1] == second_Point[1]: ## 회전하지 않은 경우
#     if  ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) < ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2): # height > width 
#         angle_point = 90
#     elif ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) > ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):
#         angle_point = 0
#     else : # height == width 
#         angle_point = 0
    
# angle_point = round(angle_point,1)

# print(f"angle_point {angle_point}")

# cv2.imshow('Corners', img)
# cv2.imshow('Corners2', im0)
# cv2.waitKey()
# cv2.destroyAllWindows()


