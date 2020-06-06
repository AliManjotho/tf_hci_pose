from src.core.point2d import Point2D
from src.core.limb import Limb
from src.utils.paf import getPAFMapImage
import cv2

joint1 = Point2D(205,260)
joint2 = Point2D(160,360)
lmb = Limb(joint1, joint2)

img = cv2.imread("images\person.jpg")
pafmap = getPAFMapImage(img, lmb)

cv2.imshow("PAF", pafmap)
key = cv2.waitKey(0)

if key == 27:
    exit(0)

