from src.dataset.cocodataset import COCODataset
import cv2


#coco17 = COCODataset()
#coco17.load()
#trainDataInstances = coco17.getTrainInstances()

#for instance in trainDataInstances:
#    print(instance)
#    img = instance.getAnnotatedImage()
#    cv2.imshow('Image', img)
#   cv2.waitKey(0)


coco17 = COCODataset()
coco17.load()
trainDataInstances = coco17.getTrainInstances()

for instance in trainDataInstances:
    img = instance.getAnnotatedImage()
    cv2.imshow('Image', img)
    cv2.waitKey(0)
