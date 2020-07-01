import cv2
from src.utils.heatmap import getHeatmaps
from src.visualization.visualize import visualizeAllHeatmap, visualizeBackgroundHeatmap


keypoints = [ [[100, 100, 2], [105,105, 2]] ]
image = cv2.imread('images/person.jpg')

hmaps = getHeatmaps(image, keypoints, 7)

visualizeAllHeatmap(image, hmaps)
visualizeBackgroundHeatmap(image, hmaps)
cv2.waitKey(0)
