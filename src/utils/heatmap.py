import numpy as np
import cv2
from src.utils.utils import gaussian, gaussian_OP


def getHeatmap_OP(image, joint, sigma):

    height = image.shape[0]
    width = image.shape[1]

    hmap = np.zeros((height, width))

    for y in range(0, height):
        for x in range(0, width):
            hmap[y, x] = gaussian_OP(x, y, joint[0], joint[1], sigma)

    return hmap


def getHeatmapImage_OP(image, joint, sigma):

    hmap = getHeatmap_OP(image, joint, sigma)
    hmap = np.uint8(hmap * 255)
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    imgOut = cv2.addWeighted(image, 0.5, hmap, 0.5, 0)

    return imgOut





def getHeatmap(image, joint, sigmax, sigmay):

    height = image.shape[0]
    width = image.shape[1]

    hmap = np.zeros((height, width))

    for y in range(0, height):
        for x in range(0, width):
            hmap[y, x] = gaussian(x, y, joint[0], joint[1], sigmax, sigmay)

    return hmap


def getHeatmapImage(image, joint, sigmax, sigmay):

    hmap = getHeatmap(image, joint, sigmax, sigmay)
    hmap = np.uint8(hmap * 255)
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    imgOut = cv2.addWeighted(image, 0.5, hmap, 0.5, 0)

    return imgOut



#sigma = 5
#sigmax = 20
#sigmay = 20
#joint = (205, 270)
#img = cv2.imread("images\person.jpg")
#hmap = getHeatmap_OP(img, joint, sigma)
#hmapImage = getHeatmapImage_OP(img, joint, sigma)
#hmapImage2 = getHeatmapImage(img, joint, sigmax, sigmay)

#print(hmap)
#cv2.imshow("Heatmap Image", hmapImage2)
#key = cv2.waitKey(0)

#if key == 27:
#    exit(0)

