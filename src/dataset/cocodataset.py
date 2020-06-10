import numpy as np
import csv
import pandas as pd
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
        self.__valCSVFile = 'datasets/coco2017_refined/annotations/val2017.csv'
        self.__valImagesDir = 'datasets/coco2017_refined//val2017/'


    def loadTrain(self):

        csvFile = open(self.__trainCSVFile)
        csvReader = csv.reader(csvFile, delimiter=',')

        # Load DataInstances
        self.__trainDataInstances = list()

        rowCount = 0
        for row in csvReader:

            #First row is headers
            if rowCount == 0:
                rowCount += 1
            else:
                imageId = row[0]
                fileName = row[1]
                filePath = row[2]
                width = int(row[3])
                height = int(row[4])
                keypoints = self.__str2keypointList(row[5])

                self.__trainDataInstances.append( DataInstance(imageId, fileName, filePath, width, height, keypoints) )

        csvFile.close()


    def loadVal(self):

        csvFile = open(self.__valCSVFile)
        csvReader = csv.reader(csvFile, delimiter=',')

        # Load DataInstances
        self.__valDataInstances = list()

        rowCount = 0
        for row in csvReader:

            #First row is headers
            if rowCount == 0:
                rowCount += 1
            else:
                imageId = row[0]
                fileName = row[1]
                filePath = row[2]
                width = int(row[3])
                height = int(row[4])
                keypoints = self.__str2keypointList(row[5])

                self.__valDataInstances.append( DataInstance(imageId, fileName, filePath, width, height, keypoints) )

        csvFile.close()




    def getTrainInstances(self):
        return self.__trainDataInstances

    def getValidationInstances(self):
        return self.__valDataInstances


    def getTrainInstancesAsPD(self):

        dataList = list()

        for instance in self.__trainDataInstances:
            dataList.append([instance.getImageId(), instance.getImageFileName(), instance.getImagePath(), instance.getImageWidth(), instance.getImageHeight(), instance.getKeypoints()])

        return  pd.DataFrame(dataList, columns=['Id', 'ImageName', 'ImagePath', 'Width', 'Height', 'Keypoints'])

    def getValidationInstancesAsPD(self):

        dataList = list()

        for instance in self.__valDataInstances:
            dataList.append([instance.getImageId(), instance.getImageFileName(), instance.getImagePath(), instance.getImageWidth(), instance.getImageHeight(), instance.getKeypoints()])

        return  pd.DataFrame(dataList, columns=['Id', 'ImageName', 'ImagePath', 'Width', 'Height', 'Keypoints'])


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





