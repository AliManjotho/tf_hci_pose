import numpy as np
import os
import shutil
import csv
from pycocotools.coco import COCO
from src.utils.utils import keypoints2String


class COCORefiner:

    def __init__(self):
        self.__trainAnnFile = 'datasets/coco2017/annotations/person_keypoints_train2017.json'
        self.__valAnnFile = 'datasets/coco2017/annotations/person_keypoints_val2017.json'
        self.__trainImagesDir = 'datasets/coco2017/train2017/'
        self.__valImagesDir = 'datasets/coco2017/val2017/'

        self.MIN_KEYPOINTS = 5
        self.MIN_AREA = 32 * 32


    def generateRefinedDataset(self, type='train', outputRootDir=''):

        self.__type = type

        if self.__type == 'train':
            self.__coco = COCO(self.__trainAnnFile)
        elif self.__type == 'val':
            self.__coco = COCO(self.__valAnnFile)

        # Create root directory if not exists
        if not os.path.isdir(outputRootDir):
            os.mkdir(outputRootDir)

        outputDir = outputRootDir + '/{0}2017'.format(self.__type)
        annDir = outputRootDir + "/annotations"

        # Create output directory if not exists
        if not os.path.isdir(outputDir):
            os.mkdir(outputDir)

        # Create annotation directory if not exists
        if not os.path.isdir(annDir):
            os.mkdir(annDir)

        # Create CSV annotation file
        csvFile = open(annDir + '/{0}2017.csv'.format(self.__type), 'w', newline='')
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["ID", "FileName", "FilePath", "Width", "Height", "Keypoints"])


        # Get image ids from annotation, containing only people
        self.__catIds = self.__coco.getCatIds(catNms=['person'])
        self.__imgIds = self.__coco.getImgIds(catIds=self.__catIds)

        images = self.__coco.loadImgs(self.__imgIds)

        # Iterate through each image
        imageCount = 0
        for image in images:

            imageFileName = image['file_name']
            imagePath = ''

            if self.__type == 'train':
                imagePath = self.__trainImagesDir + image['file_name']
            elif self.__type == 'val':
                imagePath = self.__valImagesDir + image['file_name']

            imageWidth = image['width']
            imageHeight = image['height']

            # Fetch annotations from Ann file
            annIds = self.__coco.getAnnIds(imgIds=image['id'], catIds=self.__catIds, iscrowd=None)
            anns = self.__coco.loadAnns(annIds)

            keypoints = list()

            # Populate keypoints fron annotations
            for ann in anns:

                nKeypoints = ann['num_keypoints']
                area = ann['area']

                if nKeypoints < self.MIN_KEYPOINTS or area < self.MIN_AREA:
                    continue

                kps = ann['keypoints']
                xs = np.array([kps[x] for x in range(0, len(kps), 3)])
                ys = np.array([kps[y] for y in range(1, len(kps), 3)])
                vs = np.array([kps[v] for v in range(2, len(kps), 3)])

                # Only keep those keypoints where atleat one point is set visible
                if np.any(vs > 0):
                    keypoints.append(list(zip(xs, ys, vs)))

            # Skip the image if none of the keypoint is visible for any person
            if not keypoints:
                continue

            imageCount = imageCount + 1


            # Generate meta info for new image to be generated
            newImageId = '{0:010d}'.format(imageCount)
            newImageFileName = newImageId + ".jpg"
            newImageFilePath = outputDir + "/" + newImageFileName
            shutil.copyfile(imagePath, newImageFilePath)

            # Write a data instance in to CSV file
            csvWriter.writerow([newImageId, newImageFileName, newImageFilePath, imageWidth, imageHeight, keypoints2String(np.array(keypoints))])

            print('{0} copied of {1}'.format(imageCount, len(images)))

        csvFile.close()