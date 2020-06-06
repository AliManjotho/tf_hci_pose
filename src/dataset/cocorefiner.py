import numpy as np
import os
import shutil
import csv
from pycocotools.coco import COCO


class COCORefiner:

    def __init__(self):
        self.__trainAnnFile = 'datasets/coco2017/annotations/person_keypoints_train2017.json'
        self.__valAnnFile = 'datasets/coco2017/annotations/person_keypoints_val2017.json'
        self.__trainImagesDir = 'datasets/coco2017/train2017/'
        self.__testImagesDir = 'datasets/coco2017/test2017/'
        self.__valImagesDir = 'datasets/coco2017/val2017/'
        self.__coco_train = COCO(self.__trainAnnFile)


    def generateRefinedTrainingSet(self, outputRootDir):

        # Create root directory if not exists
        if not os.path.isdir(outputRootDir):
            os.mkdir(outputRootDir)

        outputDir = outputRootDir + "/train2017"
        annDir = outputRootDir + "/annotations"

        # Create output directory if not exists
        if not os.path.isdir(outputDir):
            os.mkdir(outputDir)

        # Create annotation directory if not exists
        if not os.path.isdir(annDir):
            os.mkdir(annDir)

        # Create CSV annotation file
        csvFile = open(annDir + '/train2017.csv', 'w', newline='')
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["ID", "FileName", "FilePath", "Width", "Height", "Keypoints"])


        # Get image ids from annotation, containing only people
        self.__train_catIds = self.__coco_train.getCatIds(catNms=['person'])
        self.__train_imgIds = self.__coco_train.getImgIds(catIds=self.__train_catIds)

        images = self.__coco_train.loadImgs(self.__train_imgIds)

        # Iterate through each image
        imageCount = 0
        for image in images:

            imageCount = imageCount + 1

            imageFileName = image['file_name']
            imagePath = self.__trainImagesDir + image['file_name']
            imageWidth = image['width']
            imageHeight = image['height']

            # Fetch annotations from Ann file
            annIds = self.__coco_train.getAnnIds(imgIds=image['id'], catIds=self.__train_catIds, iscrowd=None)
            anns = self.__coco_train.loadAnns(annIds)

            keypoints = list()

            # Populate keypoints fron annotations
            for ann in anns:
                kps = ann['keypoints']
                xs = np.array([kps[x] for x in range(0, len(kps), 3)])
                ys = np.array([kps[y] for y in range(1, len(kps), 3)])
                vs = np.array([kps[v] for v in range(2, len(kps), 3)])

                # Only keep those keypoints where atleat one point is set visible
                if np.any(vs==2):
                    keypoints.append(list(zip(xs, ys, vs)))

            # Skip the image if none of the keypoint is visible for any person
            if not keypoints:
                continue

            # Generate meta info for new image to be generated
            newImageId = '{0:010d}'.format(imageCount)
            newImageFileName = newImageId + ".jpg"
            newImageFilePath = outputDir + "/" + newImageFileName
            shutil.copyfile(imagePath, newImageFilePath)

            # Write a data instance in to CSV file
            csvWriter.writerow([newImageId, newImageFileName, newImageFilePath, imageWidth, imageHeight, self.__keypointList2str(keypoints)])

            print('{0} copied of {1}'.format(imageCount, len(images)))

        csvFile.close()


    def __keypointList2str(self, keypoints):

        keypointsStr = ''
        for person in keypoints:
            for kp in person:
                keypointsStr = keypointsStr + str(kp[0]) + ',' + str(kp[1]) + ','  + str(kp[2]) + '\n'

            keypointsStr = keypointsStr + ';\n'

        keypointsStr = keypointsStr.replace('\n;', ';')
        keypointsStr = keypointsStr[:-2]

        return keypointsStr