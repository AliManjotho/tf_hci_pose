import numpy as np
from src.utils.utils import gaussian


def getHeatmaps(image, keypoints, sigma):

    height = image.shape[0]
    width = image.shape[1]
    nJoints = 17

    hmaps = np.zeros((nJoints + 1, height, width))

    # Iterate through keypoints of each person in image
    for person_keypoints in keypoints:

        for jointIndex in range(nJoints):
            kps = person_keypoints[jointIndex]

            if kps[2] > 0:
                p_x, p_y = np.meshgrid(np.arange(width), np.arange(height))
                hmaps[jointIndex] = np.maximum(hmaps[jointIndex], gaussian(p_x, p_y, kps[0], kps[1], sigma))

    # Background heatmap
    hmaps[nJoints] = 1 - np.sum(hmaps[0:nJoints], axis=0)

    #heatmap[:, :, -1] = np.clip(1.0 - np.amax(heatmap, axis=2), 0.0, 1.0)

    return hmaps