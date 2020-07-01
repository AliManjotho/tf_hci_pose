import numpy as np
from src.utils.utils import gaussian


def getHeatmaps(image, keypoints, sigma):

    height = image.shape[0]
    width = image.shape[1]
    nJoints = 17

    hmaps = np.zeros((nJoints + 1, height, width))

    # Iterate through keypoints of each person in image
    for person_keypoints in keypoints:

        for jointIndex, joint in enumerate(person_keypoints):
            kps = joint

            if kps[2] > 0:
                p_x, p_y = np.meshgrid(np.arange(width), np.arange(height))
                hmaps[jointIndex] = np.maximum(hmaps[jointIndex], gaussian(p_x, p_y, kps[0], kps[1], sigma))

    # Background heatmap
    hmaps[nJoints] = 1 - np.sum(hmaps[0:nJoints], axis=0)

    return hmaps