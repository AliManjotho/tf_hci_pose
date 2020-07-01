import numpy as np
import cv2
from src.utils.constants import CONSTANTS
from src.core.color import Color

def visualizeAllHeatmap(image, hmaps, winTitle='Heatmaps'):
    img = image.copy()

    # Get all heatmap except background
    hmaps = hmaps[0:-2]
    hmaps = hmaps.max(axis=0)
    hmaps = np.uint8(hmaps / hmaps.max() * 255)

    hmaps = cv2.applyColorMap(hmaps, cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.5, hmaps, 0.5, 0)
    cv2.imshow(winTitle, img)

    return img


def visualizeBackgroundHeatmap(image, hmaps, winTitle='Background Heatmap'):
    img = image.copy()

    # Get only background heatmap
    hmaps = hmaps[-1]
    hmaps = np.uint8(hmaps / hmaps.max() * 255)

    hmaps = cv2.applyColorMap(hmaps, cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.5, hmaps, 0.5, 0)
    cv2.imshow(winTitle, img)

    return img


def visualizePAF(img, pafs, showLimb=-1, winTitle='PAFs', type='arrows'):
    img = img.copy()

    if showLimb == -1:
        start = 0
        end = pafs.shape[0]
    else:
        start = showLimb
        end = showLimb + 1

    for i in range(start, end):
        paf_x = pafs[i,0,:,:]
        paf_y = pafs[i,1,:,:]
        len_paf = np.sqrt(paf_x**2 + paf_y**2)

        if type == 'arrows':
            step = 8
        elif type == 'circles':
            step = 4

        for x in range(0,img.shape[0],step):
            for y in range(0, img.shape[1], step):
                if len_paf[x,y]>0.25:
                    if type == 'arrows':
                        img = cv2.arrowedLine(img, (y,x), (int(y + 1*paf_x[x,y]), int(x + 6*paf_y[x,y])), CONSTANTS.colorPalatte[i], 1, cv2.LINE_AA, tipLength=1)
                    elif type == 'circles':
                        img = cv2.circle(img, (y, x), 1, CONSTANTS.colorPalatte[i], 1)
    cv2.imshow(winTitle, img)

    return img


def visualizeSkeleton(img, keypoints, winTitle='Skeleton'):
    image = img.copy()
    height = img.shape[0]
    width = img.shape[1]

    boneColor = Color.GREEN
    jointColor = Color.YELLOW

    overlay = img.copy()

    for person_keypoints in keypoints:

        limbs = [(0, 1), (0, 2), (1, 3), (2, 4),
                 (5, 7), (7, 9),
                 (6, 8), (8, 10),
                 (5, 6),
                 (11, 12),
                 (11, 13), (13, 15),
                 (12, 14), (14, 16)]

        nose = person_keypoints[0]
        lshd = person_keypoints[5]
        rshd = person_keypoints[6]
        lhip = person_keypoints[11]
        rhip = person_keypoints[12]

        neck = (int((lshd[0] + rshd[0]) / 2), int((lshd[1] + rshd[1]) / 2))
        mhip = (int((lhip[0] + rhip[0]) / 2), int((lhip[1] + rhip[1]) / 2))

        for limb in limbs:
            if person_keypoints[limb[0]][2] > 0 and person_keypoints[limb[1]][2] > 0:
                cv2.line(overlay, (person_keypoints[limb[0]][0],person_keypoints[limb[0]][1]), (person_keypoints[limb[1]][0],person_keypoints[limb[1]][1]), color=boneColor, thickness=5, lineType=cv2.LINE_AA )

        if nose[2] > 0 and lshd[2] > 0 and rshd[2] > 0:
            cv2.line(overlay, (nose[0], nose[1]), neck, color=boneColor, thickness=5, lineType=cv2.LINE_AA )

        if lshd[2] > 0 and rshd[2] > 0 and lhip[2] > 0 and rhip[2] > 0:
            cv2.line(overlay, neck, mhip, color=boneColor, thickness=5, lineType=cv2.LINE_AA )

        for kp in person_keypoints:
            cv2.circle(overlay, (kp[0],kp[1]), radius=5, color=jointColor, thickness=-1, lineType=cv2.LINE_AA)

    image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    cv2.imshow(winTitle, image)

    return image


