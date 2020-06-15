import cv2
import numpy as np


img = np.zeros((480,640,3), np.uint8)
img = cv2.arrowedLine(img, (50,50), (60,60), (0,0,255), thickness=1, line_type=cv2.LINE_AA, tipLength=0.5)


cv2.imshow('Arrow', img)
cv2.waitKey(0)