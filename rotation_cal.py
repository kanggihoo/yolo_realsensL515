import math

#3# 꼭짓점 주변 각도 변화로 찾기
import cv2
import numpy as np

# img = cv2.imread("./tmp4.png", cv2.IMREAD_GRAYSCALE)

## 다각형 근사화
# img = cv2.blur(img , (5,5))
# # img = cv2.medianBlur(img , 5)
# img2 = cv2.dilate(img , (3,3) ,iterations=5)
# canny = cv2.Canny(img2 , 50 , 200 )
# contours , _ = cv2.findContours(canny, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE )
# rectangles = []
# for contour in contours:
#     # cv2.drawContours(img , [contour] , -1 , (255,255,255) , 2)
#     # # 사각형의 꼭짓점을 추정합니다.
#     approx = cv2.approxPolyDP(contour, 1 * cv2.arcLength(contour, True), True)
#     rectangles.append(approx)
# print(rectangles)
#     # if len(approx) == 4:
#     #     rectangles.append(approx)

# cv2.drawContours(img2 , rectangles, -1 , (255,255,255) , 5)
    

## 코너 검출
# max_corners = 100  # 검출할 최대 코너점 개수
# quality_threshold = 0.01  # 코너점으로 인정되는 최소 품질 값 (0 ~ 1)
# min_distance = 5  # 코너점들 간의 최소 거리

# # Shi-Tomasi 코너 검출
# corners = cv2.goodFeaturesToTrack(img, max_corners, quality_threshold, min_distance)

# # 코너점 좌표를 정수형으로 변환하여 표시
# corners = np.array(np.intp(corners))
# # for corner in corners:
# #     x, y = corner.ravel()
# #     cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
# tmp = np.zeros_like(img)
# # cv2.drawContours(tmp , corners , -1 , (255,255,255) , 2)

# # 결과 이미지 출력
# cv2.imshow("Corners", img)
# cv2.imshow("Corners2", tmp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


## 직선검출


# cv2.imshow("canny" , canny)
# cv2.imshow("img" , img)
# cv2.imshow("img2" , img2)
# cv2.waitKey()
# cv2.destroyAllWindows()




# ##
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
 
 
 


image = cv2.imread('img/tmp4.jpg', 0)

# 이미지 이진화
_, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# 외곽선 검출
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 가장 큰 외곽선 추출
largest_contour = max(contours, key=cv2.contourArea)

image = cv2.cvtColor(image , cv2.COLOR_GRAY2RGB)
cv2.drawContours(image, [largest_contour] , -1 , (255,0,0) , 2 )
c = np.array(largest_contour).squeeze()
x = c[:,0]
y = c[:,1]
x_min, y_min= min(x) , min(y)
x_max, y_max= max(x) , max(y)
x_min_indexes = np.where(x == x_min)[0]
x_max_indexes = np.where(x == x_max)[0]
y_min_indexes = np.where(y == y_min)[0]
y_max_indexes = np.where(y == y_max)[0]

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
        
    
    
    


## 방법2 근사화
image = cv2.imread('img/tmp4.jpg', 0)
_, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

image = cv2.cvtColor(image , cv2.COLOR_GRAY2RGB)
rect = cv2.minAreaRect(largest_contour)
box = cv2.boxPoints(rect)
approximate_rect = np.zeros_like(image)

box = np.intp(box)

first_Point , second_Point , third_Point , last_Point = box

center = [int(round(i,0)) for i in [np.mean(box[:,0]) , np.mean(box[:,1])]]
print(f"center : {center}")



## 회전방향 확인
direction  = None
if first_Point[1] == second_Point[1]: # 둘이 y좌표값이 같은 경우
    direction = 'NO_rotate'
elif first_Point[1] >= second_Point[1]:
    direction ='CCW'
else:
    direction ='CW'

## 가로 세로 길이 확인(pixel단위) => heigh, width는 다시 짜야 할 거 같은데

height , width = None , None
if direction == 'No_rotate':
    width = second_Point[0] - first_Point[0]
    height = third_Point[1] - second_Point[1]
elif direction == 'CW':
    height = math.sqrt( (second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2)
    width = math.sqrt( (second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2)
    cv2.line(image , second_Point , third_Point , (255,0,0),3 ) # height(blue)
    cv2.line(image , first_Point , second_Point , (0,0,255),3 ) # width(red)
elif direction =='CCW':
    height = math.sqrt( (second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2)
    width = math.sqrt( (second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2)
    cv2.line(image , first_Point , second_Point , (255,0,0) ,2) # height(blue)
    cv2.line(image , second_Point , third_Point ,(0,0,255) ,2) # width(red)
    
print(f"box_point : {box}")
print(f"height , width : {height} , {width}")
print(f"dircetion : {direction}")

# 각도 계산
angle = 0
if direction == 'NO_rotate':
    angle = 0
    if height > width:
        angle += 90
elif direction =='CW':
    angle = round(math.atan2(-(second_Point[1] - first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi)
elif direction =='CCW':
    angle =  90 -round(math.atan2(-(second_Point[1] - first_Point[1]) , second_Point[0] - first_Point[0])*180/math.pi)
print(f"angle : {angle}")
    
## 중심점 시각화

cv2.circle(image , center , 3 , (255,0,0) , -1)
cv2.drawContours(approximate_rect , [box] , -1 , (255,0,0) , 3 )

cv2.imshow("image" , image)
cv2.imshow("image2" , approximate_rect)

cv2.waitKey()
cv2.destroyAllWindows()

