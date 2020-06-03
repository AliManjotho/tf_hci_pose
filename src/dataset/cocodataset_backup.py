from pycocotools.coco import COCO
import numpy as np
from src.dataset.dataset import Dataset
from src.dataset.datainstance import DataInstance


class COCODataset(Dataset):

    def __init__(self):

        # Override base class attributes
        self._name = "COCO 2017"
        self._nKeypoints = 16

        # Define annotation and image directories
        self.__trainAnnFile = 'datasets/coco2017/annotations/person_keypoints_train2017.json'
        self.__valAnnFile = 'datasets/coco2017/annotations/person_keypoints_val2017.json'
        self.__trainImagesDir = 'datasets/coco2017/train2017/'
        self.__testImagesDir = 'datasets/coco2017/test2017/'
        self.__valImagesDir = 'datasets/coco2017/val2017/'

        # Other attributes

        #self.__train_catIds
        #self.__train_imgIds
        #self.__val_catIds
        #self.__val_imgIds
        #self.__trainDataInstances


    def load(self):

        # Create cocoapi objects
        self.coco_train = COCO(self.__trainAnnFile)
        self.coco_val = COCO(self.__valAnnFile)

        # Get image ids from annotation, containing only people
        self.__train_catIds = self.coco_train.getCatIds(catNms=['person'])
        self.__val_catIds = self.coco_val.getCatIds(catNms=['person'])
        self.__train_imgIds = self.coco_train.getImgIds(catIds=self.__train_catIds)
        self.__val_imgIds = self.coco_val.getImgIds(catIds=self.__val_catIds)

        # Count number of train, validation ans test images
        self._numTrainImages = len(self.__train_imgIds)
        self._numValImages = len(self.__val_imgIds)
        self._numTestImages = 0

        # Load TrainDataInstances
        self.__trainDataInstances = list()
        images = self.coco_train.loadImgs(self.__train_imgIds)

        #Iterate through all images
        for image in images:

            #Get single train dataset instance
            fileName = image['file_name']
            path = self.__trainImagesDir + image['file_name']
            w = image['width']
            h = image['height']

            # Fetch annotations from Ann file
            annIds = self.coco_train.getAnnIds(imgIds=image['id'], catIds=self.__train_catIds, iscrowd=None)
            anns = self.coco_train.loadAnns(annIds)

            keypoints = list()

            # Populate keypoints fron annotations
            for ann in anns:
                kps = ann['keypoints']
                xs = np.array([kps[x] for x in range(0, len(kps), 3)])
                ys = np.array([kps[y] for y in range(1, len(kps), 3)])
                vs = np.array([kps[v] for v in range(2, len(kps), 3)])
                keypoints.append( list(zip(xs, ys, vs)) )

            self.__trainDataInstances.append(DataInstance(fileName, path, w, h, keypoints))


    def getTrainInstances(self):
        return self.__trainDataInstances





