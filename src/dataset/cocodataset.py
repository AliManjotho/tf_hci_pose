import numpy as np
import csv
from src.dataset.dataset import Dataset
from src.dataset.datainstance import DataInstance


class COCODataset(Dataset):

    def __init__(self):

        # Override base class attributes
        self._name = "COCO 2017"
        self._nKeypoints = 16

        # Define annotation and image directories
        self.__trainCSVFile = 'datasets/coco2017_refined/annotations/train2017.csv'
        self.__trainImagesDir = 'datasets/coco2017_refined//train2017/'


    def load(self):

        csvFile = open(self.__trainCSVFile)
        csReader = csv.reader(csvFile, delimiter=',')

        # Load TrainDataInstances
        self.__trainDataInstances = list()

        rowCount = 0
        for row in csReader:

            #First row is headers
            if rowCount == 0:
                rowCount += 1
            else:
                imageId = row[0]
                fileName = row[1]
                filePath = row[2]
                width = row[3]
                height = row[4]
                keypoints = self.__str2keypointList(row[5])

                self.__trainDataInstances.append( DataInstance(imageId, fileName, filePath, width, height, keypoints) )

        csvFile.close()


    def getTrainInstances(self):
        return self.__trainDataInstances



    def __str2keypointList(self, keypointStr):

        keypoints = list()
        personsStr = keypointStr.split(';\n')

        for personStr in personsStr:
            kpsStr = personStr.split('\n')

            personKeypoints = list()
            for kpStr in kpsStr:
                kpPointsStr = kpStr.split(',')

                kpTuple = (int(kpPointsStr[0]), int(kpPointsStr[1]), int(kpPointsStr[2]))
                personKeypoints.append(kpTuple)

            keypoints.append(personKeypoints)

        return keypoints





