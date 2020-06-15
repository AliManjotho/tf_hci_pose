from src.dataset.cocodataset import COCODataset
import cv2

# Training Data Inspect

coco17 = COCODataset()
coco17.loadTrain()
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






# Validation Data Inspect

# coco17 = COCODataset()
# coco17.loadVal()
# valDataInstances = coco17.getValidationInstances()
#
# index = 0
#
# while True:
#     instance = valDataInstances[index]
#     img = instance.getAnnotatedImage()
#     cv2.imshow('Image', img)
#     key = cv2.waitKey(0)
#
#     if key == 27:
#         exit(0)
#     elif key == 115:
#         index = index + 1
#     elif key == 97:
#         index = index - 1
#
#     if index < 0 or index >= len(valDataInstances):
#         break
#
# cv2.destroyAllWindows()