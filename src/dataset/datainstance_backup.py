import numpy as np
import cv2

class DataInstance:

    def __init__(self, imageFileName, imagePath, imageWidth, imageHeight, keypoints):
        self.__imageFileName = imageFileName
        self.__imagePath = imagePath
        self.__imageWidth = imageWidth
        self.__imageHeight= imageHeight
        self.__keypoints = keypoints
        self.__heatmaps = 0
        self.__pafs = 0

    def getImageFileName(self):
        return self.__imageFileName

    def getImagePath(self):
        return self.__imagePath

    def getImageWidth(self):
        return self.__imageWidth

    def getImageHeight(self):
        return self.__imageHeight

    def getKeypoints(self):
        return self.__keypoints

    def getImage(self):
        image = cv2.imread(self.__imagePath)
        return image

    def getResizedImage(self, newWidth, newHeight):
        image = self.getImage()
        resizedImage = cv2.resize(image, (newWidth, newHeight), interpolation = cv2.INTER_AREA)
        return resizedImage


    def getAnnotatedImage(self):
        image = self.getImage()
        annotedImage = self.__drawSkeleton(image)
        return annotedImage



    def getAnnotatedResizedImage(self, newWidth, newHeight):
        image = self.getResizedImage(newWidth, newHeight)
        annotedImage = self.__drawSkeleton(image, True, newWidth, newHeight)
        return annotedImage


    def __drawSkeleton(self, image, resized=False, newWidth=1, newHeight=1):

        joints = {  0 : "Nose",
                    1 : "LeftEye",
                    2 : "RightEye",
                    3 : "LeftEar",
                    4 : "RightEar",
                    5 : "LeftShoulder",
                    6 : "RightShoulder",
                    7 : "LeftElbow",
                    8 : "RightElbow",
                    9 : "LeftWrist",
                   10 : "RightWrist",
                   11 : "LeftHip",
                   12 : "RightHip",
                   13 : "LeftKnee",
                   14 : "RightKnee",
                   15 : "LeftAnkle",
                   16 : "RightAnkle"  }

        limbs = [(0,1), (0,2), (1,3), (2,4),
                 (3,5), (5,7), (7,9),
                 (4,6), (6,8), (8,10),
                 (5, 6),
                 (11,12),
                 (5,11), (11,13), (13,15),
                 (6,12), (12,14), (14, 16)]


        jointColor = (0, 0, 255)
        boneColor =  (0, 255, 0)

        for person in self.getKeypoints():

            for limb in limbs:

                x1 = y1 = x2 = y2 = 0

                if resized:
                    x1 = int ( person[limb[0]][0] / self.__imageWidth * newWidth )
                    y1 = int ( person[limb[0]][1] / self.__imageHeight * newHeight )
                    x2 = int ( person[limb[1]][0] / self.__imageWidth * newWidth )
                    y2 = int ( person[limb[1]][1] / self.__imageHeight * newHeight )
                else:
                    x1 = person[limb[0]][0]
                    y1 = person[limb[0]][1]
                    x2 = person[limb[1]][0]
                    y2 = person[limb[1]][1]

                v1 = person[limb[0]][2]
                v2 = person[limb[1]][2]

                if v1 > 0 and v2 > 0:
                    image = cv2.line(image, (x1,y1), (x2,y2), boneColor, lineType=cv2.LINE_AA )

                if v1 > 0:
                    image = cv2.circle(image, (x1, y1), 4, jointColor, thickness=-1, lineType=cv2.LINE_AA)

                if v2 > 0:
                    image = cv2.circle(image, (x2, y2), 4, jointColor, thickness=-1, lineType=cv2.LINE_AA)

        return image


    def __str__(self):
        return "Image: {0}\n" \
               "Width: {1}\n" \
               "Height: {2}\n" \
               "Num: of persons: {3}".format(self.__imageFileName, self.__imageWidth, self.__imageHeight, len(self.__keypoints))

