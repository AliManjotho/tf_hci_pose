import cv2
import PIL as Image
import numpy as np
from src.utils.paf import getPAF, LIMBS
from src.utils.heatmap import get_heatmap
from src.visualization.visualize import visualizePAF, visualizeSkeleton, visualizeHeatmap


keypoints_test_1 = [[
    (322, 183, 2),
    (333, 173, 2),
    (310, 175, 2),
    (351, 179, 2),
    (296, 185, 2),
    (378, 240, 2),
    (274, 232, 2),
    (388, 319, 2),
    (247, 286, 2),
    (403, 391, 2),
    (268, 276, 2),
    (349, 400, 2),
    (290, 398, 2),
    (339, 523, 2),
    (274, 536, 2),
    (318, 613, 2),
    (269, 636, 2)
]]

keypoints_test_2 = [
    [(307, 177, 2), (316, 169, 2), (291, 167, 2), (328, 185, 2), (268, 179, 2), (327, 235, 2), (246, 249, 2), (0, 0, 0), (219, 350, 2), (0, 0, 0), (237, 439, 2), (332, 427, 2), (281, 446, 2), (320, 544, 2), (276, 585, 2), (297, 648, 2), (285, 698, 2)],
    [(453, 182, 2), (464, 175, 2), (441, 171, 2), (472, 189, 2), (420, 177, 2), (488, 240, 2), (385, 226, 2), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (462, 402, 2), (399, 403, 2), (446, 514, 2), (400, 537, 2), (430, 611, 2), (402, 627, 2)],
    [(575, 209, 2), (585, 200, 2), (563, 200, 2), (599, 210, 2), (547, 215, 2), (632, 249, 2), (523, 242, 2), (0, 0, 0), (0, 0, 0), (761, 237, 2), (407, 199, 2), (604, 419, 2), (553, 416, 2), (599, 506, 2), (560, 522, 2), (583, 603, 2), (593, 607, 2)],
    [(693, 211, 2), (705, 200, 2), (683, 201, 2), (722, 208, 2), (673, 212, 2), (753, 260, 2), (657, 263, 2), (783, 364, 2), (0, 0, 0), (778, 431, 2), (0, 0, 0), (745, 455, 2), (690, 453, 2), (750, 547, 2), (695, 546, 2), (748, 637, 2), (709, 626, 2)]
]

image_test_1 = cv2.imread('images/test_image_1.jpg')
image_test_2 = cv2.imread('images/test_image_2.jpg')

pafs = getPAF(image_test_2, keypoints_test_2, 7)
img = visualizePAF(image_test_2, pafs, showLimb=8)
#cv2.imwrite('test_image_2_paf.jpg', img)

hmaps = get_heatmap(image_test_2, keypoints_test_2, sigma=10)
img = visualizeHeatmap(image_test_2, hmaps, limbId=-1)
#cv2.imwrite('test_image_2_hmap.jpg', img)

img = visualizeSkeleton(image_test_2, keypoints_test_2)
#cv2.imwrite('test_image_2_skeleton.jpg', img)