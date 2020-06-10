import numpy as np
from math import sqrt, pow
import cv2
from src.core.limb import Limb
from src.core.point2d import Point2D


def getPAF(image, limb : Limb):

    height = image.shape[0]
    width = image.shape[1]
    paf = np.zeros((height, width))

    for y in range(0, height):
        for x in range(0, width):
            p = Point2D(x,y)
            if isPointOnLimb(p, limb):
                paf[y, x] = getV(limb)

    return paf



def getPAFMap(image, limb : Limb):

    height = image.shape[0]
    width = image.shape[1]
    paf = np.zeros((height, width))

    for y in range(0, height):
        for x in range(0, width):
            if isPointOnLimb(Point2D(x,y), limb):
                paf[y, x] = 1

    return paf


def getPAFMapImage(image, limb):
    paf = getPAFMap(image, limb)
    paf = np.uint8(paf * 255)
    paf = cv2.applyColorMap(paf, cv2.COLORMAP_COOL)
    imgOut = cv2.addWeighted(image, 0.5, paf, 0.5, 0)

    imgOut = cv2.circle(imgOut, (limb.joint1.x, limb.joint1.y), 10, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    imgOut = cv2.circle(imgOut, (limb.joint2.x, limb.joint2.y), 10, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    return imgOut




def getPAFImage(image, limb):
    paf = getPAF(image, limb)
    paf = np.uint8(paf * 255)
    paf = cv2.applyColorMap(paf, cv2.COLORMAP_JET)
    imgOut = cv2.addWeighted(image, 0.5, paf, 0.5, 0)

    return imgOut



def getV(limb : Limb):
    j1 = limb.joint1
    j2 = limb.joint2
    denom = sqrt( pow((j2.x - j1.x), 2) + pow((j2.y - j1.y), 2) )
    return Point2D( (j2.x - j1.x) / denom, (j2.y - j1.y) /denom )

def getVPerp(limb : Limb):
    v = getV(limb)
    return Point2D(v.y, -v.x)

def isPointOnLimb(p : Point2D, limb : Limb):
     v = getV(limb)
     v_ = getVPerp(limb)
     l = limb.getLength()
     lw = 20
     j1 = limb.joint1

     val1 = v.dot(p - j1)
     val2 = abs( v_.dot(p - j1) )

     return val1 >=0 and val1 <= l and val2 <=lw

