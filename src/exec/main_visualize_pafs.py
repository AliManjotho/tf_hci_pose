import numpy as np
import cv2
from src.dataset.cocodataset import COCODataset
from src.utils.paf import getPAFs
from src.visualization.visualize import visualizePAF
from math import degrees, atan2

coco17 = COCODataset()
coco17.loadTrain()
trainDataInstances = coco17.getTrainInstances()
instance = trainDataInstances[561]

img = instance.getImage()
keypoints = instance.getKeypoints()


#pafs = getPAFs(img, keypoints, 7)
#img = visualizePAF(img, pafs, showLimb=-1, winTitle='PAFs', type='circles')
#p.savetxt('pafX.csv', pafs[8, 0, : , :], fmt="%s", delimiter=',')
#p.savetxt('pafY.csv', pafs[8, 1, : , :], fmt="%s", delimiter=',')

p1 = keypoints[0][11]
p2 = keypoints[0][13]

deltaY = (p2[1] - p1[1])
deltaX = (p2[0] - p1[0])
result = degrees(atan2(deltaY, deltaX))

if result < 0:
    result = 360 + result

print(result)
