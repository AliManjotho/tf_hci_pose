import cv2
from heatmap import getHeatmap, getHeatmap_OP, getHeatmapImage, getHeatmapImage_OP

sigma = 5
sigmax = 20
sigmay = 20
joint = (205, 270)
img = cv2.imread("images\person.jpg")
hmap = getHeatmap_OP(img, joint, sigma)
hmapImage = getHeatmapImage_OP(img, joint, sigma)
hmapImage2 = getHeatmapImage(img, joint, sigmax, sigmay)


print(hmap)
cv2.imshow("Heatmap Image", hmapImage2)
key = cv2.waitKey(0)

if key == 27:
    exit(0)