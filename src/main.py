import cv2
from src.dataset.cocodataset import COCODataset
from src.utils.paf import getPAFs
from src.utils.heatmap import getHeatmaps
from src.visualization.visualize import visualizePAF, visualizeSkeleton, visualizeAllHeatmap,visualizeBackgroundHeatmap


coco17 = COCODataset()
coco17.loadTrain()
trainDataInstances = coco17.getTrainInstances()

index = 0

while True:
    instance = trainDataInstances[index]
    image = instance.getImage()
    keypoints = instance.getKeypoints()

    pafs = getPAFs(image, keypoints, 7)
    hmaps = getHeatmaps(image, keypoints, sigma=10)

    visualizePAF(image, pafs, showLimb=-1)
    visualizeAllHeatmap(image, hmaps)
    visualizeBackgroundHeatmap(image, hmaps)
    visualizeSkeleton(image, keypoints)

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