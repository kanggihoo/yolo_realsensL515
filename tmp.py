import numpy as np
import cv2
image = cv2.imread('img/test3.jpg', 0)




def f(image):
    cv2.circle(image , (10,10) , 3 , (255,0,0) , 2)

    
cv2.imshow("aa" , image)
f(image)
cv2.imshow("aasds" , image)

cv2.waitKey()
cv2.destroyAllWindows()