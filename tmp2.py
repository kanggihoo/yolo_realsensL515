import cv2
import numpy as np
import math

def FindIOU(binary , largest_contour , box_point): # box_point : np.ndarray
    y_boundary , x_boundary = list(map(lambda x : x-1 , binary.shape))
    check_x = np.where(box_point[:,0] > x_boundary , True , False ) 
    check_y = np.where(box_point[:,1] > y_boundary , True , False)
    # print(box_point)
    # print("check_x : ", check_x,any(check_x))
    # print("check_y : ", check_y,any(check_y))
    # print(box_point)
    
    if any(check_x) or any(check_y):
        print("범위 초과!")
        box_point_max = np.max(box_point , axis = 0) 
        intersection = np.zeros(shape=(box_point_max[1] , box_point_max[0]))
    else : intersection = np.zeros_like(binary)
    
    cv2.drawContours(intersection, [largest_contour], -1, 255, thickness=cv2.FILLED)
    largest_contour_area = np.sum(intersection)/255

    intersection[:,:]=0
    cv2.drawContours(intersection, [box_point], -1, 255, thickness=cv2.FILLED)
    box_contour_area=np.sum(intersection)/255

    cv2.drawContours(intersection, [largest_contour], -1, 255, thickness=cv2.FILLED)
    union_area = np.sum(intersection)/255
    intersection_area = largest_contour_area +box_contour_area - np.sum(intersection)/255
    IOU = round(intersection_area/union_area , 2)
    # print("합집합 넓이 : ",union_area)
    # print("largest_contour 면적: " , largest_contour_area)
    # print("box contour 면적 : " , box_contour_area)
    # print("교집합 넓이:", intersection_area )
    print("IOU" , IOU)    
    return IOU

def FindFourPoint(corner_image , HEIGHT , WIDTH , exception = False ):
# corner_image processing 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    corner_image = cv2.morphologyEx(corner_image , cv2.MORPH_CLOSE , kernel=kernel , iterations=3)
    y, x = np.where(corner_image ==255)
    
    ### 최상단 점 찾기 (p1)
    min_y  = min(y)
    min_y_index = np.where(y == min_y)
    min_x = min(x[min_y_index])
    p1 = min_x , min_y

    ### x값이 가장 작은 경우(p2) y가 가장 작은 경우
    min_x  = min(x)
    min_x_index = np.where(x == min_x)
    min_y = min(y[min_x_index])
    p2 = (min_x , min_y)

    ### 경계의 하단 부이면서 x가 가장 작은 경우 (p3)
    BotomY_index = np.where(y == HEIGHT-1)
    BotomX = min(x[BotomY_index])
    p3 = (BotomX , HEIGHT-1)

    ### 경계의 우측 부이면서 y가 가장 작은 경우 (p4)
    RightX_index = np.where(x == WIDTH-1)
    RightY = min(y[RightX_index])
    p4 = (WIDTH-1 , RightY)
    return p1,p2,p3,p4

def FindLargeContour_BoxPoint(image):
    '''
    input : binary_image
    output : largest_contour , box_point 
    '''
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box_point = cv2.boxPoints(rect) 
    return  largest_contour , box_point 

def FindAngle(box_point):
    first_Point , second_Point , third_Point , last_Point = box_point
    angle = 0.0
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
            angle = 90.0
            add_90angle = True
        elif ((second_Point[0] - first_Point[0])**2 + (second_Point[1] - first_Point[1])**2) > ((second_Point[0] - third_Point[0])**2 + (second_Point[1] - third_Point[1])**2):
            angle = 0.0
        else : # height == width 
            angle = 0.0
    return round(angle,1) , add_90angle
    

def FindAngleAndLength(iamge, reference_point , depth_rect_3channel , direction, exception = False ): # 영역 예외가 발생한 경우 (가로 영역을 늘리고, 세로 영역을 늘린 경우 exception = True)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        corner_image = cv2.morphologyEx(iamge , cv2.MORPH_CLOSE , kernel=kernel , iterations=3)
        y,x = np.where(corner_image ==255) 
        if exception is True:
            print("예외 발생에 대한 end , start 지점 설정")
            end = int(depth_rect_3channel.shape[1]*0.5)
            start = int((end - reference_point[0])*0.6) + reference_point[0]
        else :
            if direction == "DOWN":
                if reference_point[0] > depth_rect_3channel.shape[1] / 4:
                    print("reference_point[0] > depth_rect_3channel.shape[1] / 4")
                    end = int(depth_rect_3channel.shape[1]*0.8)
                    # end = depth_rect_3channel.shape[1]
                    start = int((end - reference_point[0])*0.6) + reference_point[0]
                else:
                    print("reference_point[0] < depth_rect_3channel.shape[1] / 4")
                    end = int(depth_rect_3channel.shape[1]*0.7)
                    start = int((end - reference_point[0])*0.8) + reference_point[0]
            else :# direction : UP
                end = int(depth_rect_3channel.shape[1]*0.5)
                start = int((end - reference_point[0])*0.6) + reference_point[0]
                

       
        candidate = range(start , end ,2)   
        final_angle = 0 
        candidate_list = [reference_point[1] if direction== 'UP' else reference_point[0]]
        cv2.circle(depth_rect_3channel , reference_point , 3 , (0,0,255) , -1)
        for idx , i in enumerate(candidate):
            RightX_index = np.where(x == i)
            if len(RightX_index)==0:
                print("i값에 해당되는 RightX_index 없음")
                break
            RightY = min(y[RightX_index])
            RightPoint = (i , RightY)
            
            ## down에 대한 범위 벗어난 경우를 어떻게 처리하면 될까나?
            RightY_index = np.where(y==(RightY))
            next_X = max(x[RightY_index])
            # print("현재 좌표 : " , RightPoint)
            # print("다음 X값:" , next_X)
            
            if direction =='UP':
                if candidate_list[-1] >= RightY:
                    candidate_list.append(RightY) 
                else:
                    continue
            elif direction =='DOWN':
                if True: 
                    # print(candidate_list[-1] , next_X)
                    candidate_list.append(next_X)
                else:
                    print("다음 X값이 이전보다 작아집니다. ")
                    continue
            angle = math.atan2(-(RightPoint[1] - reference_point[1]) , RightPoint[0] - reference_point[0])*180/math.pi
            print(angle)
            cv2.circle(depth_rect_3channel , RightPoint , 3 , (255,0,0) , -1)
            final_angle += angle
        assert len(candidate)-1 >=1 , print("적합한 후보 없음")
        # if len(candidate_list)-1 <= 1:
        #     ##############################################################################################################후보 갯수 부족한 경우 (start 범위 더 늘림)
        #     print("후보 갯수 부족! ")
        #     if reference_point[0] > depth_rect_3channel.shape[1] / 4:
        #         print("reference_point[0] > depth_rect_3channel.shape[1] / 4")
        #         end = int(depth_rect_3channel.shape[1]*0.8)
        #         # end = depth_rect_3channel.shape[1]
        #         start = int((end - reference_point[0])*0.5) + reference_point[0]
        #     else : 
        #         print("reference_point[0] < depth_rect_3channel.shape[1] / 4")
        #         end = int(depth_rect_3channel.shape[1]*0.7)
        #         start = int((end - reference_point[0])*0.7) + reference_point[0]
        #     candidate = range(start , end ,2)   
        #     final_angle = 0 
        #     candidate_list = [reference_point[1] if direction== 'UP' else reference_point[0]]
        #     cv2.circle(depth_rect_3channel , reference_point , 3 , (0,0,255) , -1)
        #     for idx , i in enumerate(candidate):
        #         RightX_index = np.where(x == i)
        #         if len(RightX_index)==0:
        #             print("i값에 해당되는 RightX_index 없음")
        #             break
        #         RightY = min(y[RightX_index])
        #         RightPoint = (i , RightY)
                
        #         ## down에 대한 범위 벗어난 경우를 어떻게 처리하면 될까나?
        #         RightY_index = np.where(y==(RightY-5))
        #         # next_X = max(x[RightY_index])
        #         # print("현재 좌표 : " , RightPoint)
        #         # print("다음 X값:" , next_X)
                
        #         if direction =='UP':
        #             if candidate_list[-1] >= RightY:
        #                 candidate_list.append(RightY) 
        #             else:
        #                 continue
        #         # elif direction =='DOWN':
        #         #     if i > next_X:
        #         #         print(candidate_list[-1] , next_X)
        #         #         candidate_list.append(next_X)
        #         #     else:
        #         #         print("다음 X값이 이전보다 작아집니다. ")
        #         #         continue
        #         angle = math.atan2(-(RightPoint[1] - reference_point[1]) , RightPoint[0] - reference_point[0])*180/math.pi
        #         # print(angle)
        #         cv2.circle(depth_rect_3channel , RightPoint , 2 , (255,0,0) , -1)
        #         final_angle += angle
        final_angle /= (len(candidate_list)-1)
        print("total후보 갯수 : " , len(candidate) , "선정 후보 갯수" ,len(candidate_list)-1 , candidate_list)
        print("mean angle : " , final_angle)
        if final_angle <0:
            ################################# 여기서 self.spare_roi와 refrence_point[1] 중 작은 값으로 spare값으로 설정해보는 것도 좋을 듯?? 
            print(f"mean_angle : {final_angle} , reference_point : {reference_point} , reference_point[1] : {reference_point[1]} , spare_roi :{spare_roi}")
            
            spare = reference_point[1]
  
            new_slice = depth_rect_3channel[spare:depth_rect_3channel.shape[0]-spare , spare: depth_rect_3channel.shape[1]-spare].copy()
            
            reference_point = (lambda x : x-reference_point[1])(reference_point)
            
            RightSidePoint = new_slice.shape[1]-1 , ((new_slice.shape[1]-1) - reference_point[0])*math.tan(math.radians(-final_angle)) 
            RightSidePoint = np.int16(RightSidePoint)
            LeftSidePoint = 0 , (reference_point[0])*math.tan(math.radians(90-abs(final_angle))) 
            LeftSidePoint = np.int16(LeftSidePoint)
            LastPoint = math.sqrt( ((RightSidePoint[0])-reference_point[0])**2 + (RightSidePoint[1] - reference_point[1])**2)*math.cos(math.radians(-final_angle)) , new_slice.shape[0]-1
            LastPoint = np.int16(LastPoint)
            if LeftSidePoint[1] > new_slice.shape[0]:
                print("오류발생 !! , LastPoint " , LeftSidePoint , "Image.shape" , new_slice.shape)
                LeftSidePoint = spare , depth_rect_3channel.shape[0]-math.sqrt( ((RightSidePoint[0])-reference_point[0])**2 + (RightSidePoint[1] - reference_point[1])**2)*math.sin(math.radians(-final_angle))-(spare+2)
                LeftSidePoint = np.int16(LeftSidePoint)
                print(LeftSidePoint)
            print("LeftSide , LastPoint 사이 각도 : " , math.atan2(-(LastPoint[1] - LeftSidePoint[1]) , LastPoint[0] - LeftSidePoint[0])*180/math.pi)
            WidthLength =  math.sqrt( ((RightSidePoint[0])-reference_point[0])**2 + (RightSidePoint[1] - reference_point[1])**2)
            HeightLenght = math.sqrt((LeftSidePoint[0]-reference_point[0])**2 + (LeftSidePoint[1]- reference_point[1])**2)
            print("RightSide와 reference_point 길이 : " , WidthLength)
            print("LeftSide와 reference_point  길이 : " , HeightLenght)

            if WidthLength > HeightLenght:
                final_angle  = abs(final_angle)+90
            else : 
                final_angle  = abs(final_angle)
            cv2.line(new_slice , reference_point , RightSidePoint , (0,255,0) ,2)
            cv2.line(new_slice, reference_point , LeftSidePoint , (0,255,0) , 2)
            cv2.line(new_slice, LeftSidePoint , LastPoint , (0,255,0) , 2)
                
            
        else :
            ################################# 여기서 self.spare_roi와 refrence_point[0] 중 작은 값으로 spare값으로 설정해보는 것도 좋을 듯?? 
            print(f"final_angle : {final_angle} , reference_point : {reference_point} , reference_point[0] : {reference_point[0]} , spare_roi :{spare_roi}")
            
            spare = reference_point[0]
          
            new_slice = depth_rect_3channel[spare:depth_rect_3channel.shape[0]-spare , spare: depth_rect_3channel.shape[1]-spare]
            
            reference_point = (lambda x : x-reference_point[0])(reference_point)
            
            RightSidePoint = (reference_point[1]+2)/math.tan(math.radians(final_angle)) , 0
            RightSidePoint = np.int16(RightSidePoint)
            
            BotomPoint = (new_slice.shape[0]-1-reference_point[1])*math.tan(math.radians(final_angle)),  new_slice.shape[0] -1
            BotomPoint = np.int16(BotomPoint)
            
            LastPoint = math.sqrt( ((RightSidePoint[0])-reference_point[0])**2 + (RightSidePoint[1] - reference_point[1])**2)*math.cos(math.radians(final_angle))+BotomPoint[0] , (new_slice.shape[0]-1)-math.sqrt( ((RightSidePoint[0])-reference_point[0])**2 + (RightSidePoint[1] - reference_point[1])**2)*math.sin(math.radians(final_angle))
            LastPoint = np.int16(LastPoint)
            
            WidthLength =   math.sqrt( ((RightSidePoint[0])-reference_point[0])**2 + (RightSidePoint[1] - reference_point[1])**2)
            HeightLenght = math.sqrt((BotomPoint[0]-reference_point[0])**2 + (BotomPoint[1]- reference_point[1])**2)
            print("RightSide와 reference_point 길이 : " , WidthLength)
            print("BotomPoint와 reference_point  길이 : " , HeightLenght)
            
            if WidthLength > HeightLenght:
                final_angle  = final_angle
            else :
                final_angle  = final_angle+90
            cv2.line(new_slice , reference_point , RightSidePoint , (0,255,0) ,2)
            cv2.line(new_slice, reference_point , BotomPoint , (0,255,0) , 2)
            cv2.line(new_slice, BotomPoint , LastPoint , (0,255,0) , 2)
    
        return final_angle , new_slice 


image = cv2.imread("./img/tmp24.jpg")
depth_rect_one_channel = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
_, depth_rect_one_channel = cv2.threshold(depth_rect_one_channel, 128, 255, cv2.THRESH_BINARY)
depth_rect_3channel = cv2.cvtColor(depth_rect_one_channel, cv2.COLOR_GRAY2BGR)
spare_roi =3

largest_contour , box_point = FindLargeContour_BoxPoint(depth_rect_one_channel)
        ######  boxPoint 근사화 
box_int = np.intp(np.round(box_point)) # box 좌표 정수화 
IOU = FindIOU(depth_rect_one_channel , largest_contour , box_int)

if IOU > 0.9: # box_point로 찾기
    angle , add_90angle = FindAngle(box_int)
    new_slice = None
    print(angle)
else :
    ## largest_contour만을 남겨두고 나머지는 삭제 
    depth_rect_one_channel = np.zeros_like(depth_rect_one_channel)
    cv2.fillPoly(depth_rect_one_channel , [largest_contour] , 255)
    
    HEIGHT , WIDTH =  list(map(lambda x : int(x) , (depth_rect_one_channel.shape[0]*0.5 , depth_rect_one_channel.shape[1]*0.5)))
    corner_image = depth_rect_one_channel[:HEIGHT , :WIDTH] 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    p1, p2 ,p3,p4 = FindFourPoint(corner_image , HEIGHT , WIDTH )
    
    if p1[0] < p4[0] and p2[0] == p3[0]: # 좌측 상단에 최고점 존재(p1) => 구하고자 하는 각도 음수
        print("좌측 상단 최고점 p1존재")
        corner_image = cv2.morphologyEx(depth_rect_one_channel , cv2.MORPH_CLOSE , kernel=kernel , iterations=3)
        final_angle , new_slice = FindAngleAndLength(corner_image, p1 ,depth_rect_3channel , direction='DOWN')

    elif p1[1] == p4[1] and p2[0] < p3[0]: # 좌측 상단에 x값이 가장 작은 점이 있음(p2) => 구하고자 하는 각도 양수
        ## p2를 기준으로 다시 영역을 설정한 뒤 각도 계산
        print("좌측 상단 x값 가장 작음(p2)")
        corner_image = cv2.morphologyEx(depth_rect_one_channel , cv2.MORPH_CLOSE , kernel=kernel , iterations=3)
        final_angle, new_slice = FindAngleAndLength(corner_image, p2 ,depth_rect_3channel , direction='UP')
        
    elif p1[0] < p4[0]  and p2[0] != p3[0] :# tmp17 , 19.jpg 같은 경우 반전 짤랐는데 꼭짓점이 2개 있는 경우 (첫번째 case의 변형)
        print("꼭짓점 2개 발견")
        corner_image = cv2.morphologyEx(depth_rect_one_channel , cv2.MORPH_CLOSE , kernel=kernel , iterations=3)
        final_angle , new_slice = FindAngleAndLength(corner_image, p1 ,depth_rect_3channel, direction='DOWN')
    else : # 좌측을 더 넓게 탐색 
        print("예외 경우, 영역 재탐색 ")
        HEIGHT , WIDTH=  list(map(lambda x : int(x) , (depth_rect_one_channel.shape[0]*0.7 , depth_rect_one_channel.shape[1]*0.3)))
        corner_image = depth_rect_one_channel[:HEIGHT , :WIDTH]   
        p1,p2,p3,p4 = FindFourPoint(corner_image , HEIGHT , WIDTH)
        if p1[0] < p4[0] and p2[0] == p3[0]: # 좌측 상단에 최고점 존재(p1) => 구하고자 하는 각도 음수
            ## p1을 기준으로 다시 영역을 설정한 뒤 각도 계산
            print("좌측 상단 최고점 p1존재")
            corner_image = cv2.morphologyEx(depth_rect_one_channel , cv2.MORPH_CLOSE , kernel=kernel , iterations=3)
            final_angle , new_slice = FindAngleAndLength(corner_image, p1 ,depth_rect_3channel ,direction='DOWN', exception=True)

        elif p1[1] == p4[1] and p2[0] < p3[0]: # 좌측 상단에 x값이 가장 작은 점이 있음(p2) => 구하고자 하는 각도 양수
            ## p2를 기준으로 다시 영역을 설정한 뒤 각도 계산
            print("좌측 상단 x값 가장 작음(p2)")
            corner_image = cv2.morphologyEx(depth_rect_one_channel , cv2.MORPH_CLOSE , kernel=kernel , iterations=3)
            final_angle, new_slice = FindAngleAndLength(corner_image, p2 ,depth_rect_3channel,direction='UP' , exception=True)
    
    print(f"fianl_angle : {round(final_angle,1)}")

    
    





if isinstance(new_slice , np.ndarray):
    cv2.imshow("new" , new_slice)
else:
    cv2.drawContours(depth_rect_3channel , [box_int] , -1 , (255,0,0) , 2)
    cv2.drawContours(depth_rect_3channel , [largest_contour] , -1 , (0,0,255) , 2)

cv2.imshow('depth_rect_one_channel', depth_rect_one_channel)
cv2.imshow('Corners2', depth_rect_3channel)
cv2.waitKey()
cv2.destroyAllWindows()


