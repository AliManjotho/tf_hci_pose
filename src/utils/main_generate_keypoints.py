import cv2
import numpy as np


events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

refPt = []

#click event function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('({0}, {1}, 2)'.format(x, y))
        refPt.append([x,y])
        #cv2.imshow("image", img)


#Here, you need to change the image name and it's path according to your directory
img = cv2.imread('images/test_image_2.jpg')
cv2.imshow("image", img)

#calling the mouse click event
cv2.setMouseCallback("image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()



