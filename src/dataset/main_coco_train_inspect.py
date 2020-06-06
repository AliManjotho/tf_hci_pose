from src.dataset.cocodataset import COCODataset
import cv2

coco17 = COCODataset()
coco17.load()
trainDataInstances = coco17.getTrainInstances()

index = 0

while True:
    instance = trainDataInstances[index]
    img = instance.getAnnotatedImage()
    cv2.imshow('Image', img)
    key = cv2.waitKey(0)

    if key == 27:
        exit(0)
    elif key == 115:
        index = index + 1
    elif key == 97:
        index = index - 1

    if index < 0 or index >= len(trainDataInstances):
        break

cv2.destroyAllWindows()

