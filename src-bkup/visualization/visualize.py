import numpy as np
import cv2


colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]


def visualizeHeatmap(image, hmaps, limbId=0, winTitle='HMAPs'):
    img = image.copy()

    hmaps = hmaps[limbId]

    #hmaps = hmaps.max(axis=0)
    hmaps = np.uint8(hmaps / hmaps.max() * 255)

    hmaps = cv2.applyColorMap(hmaps, cv2.COLORMAP_JET)
    img = cv2.addWeighted(img, 0.5, hmaps, 0.5, 0)

    cv2.imshow(winTitle, img)
    cv2.waitKey(0)

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
                        img = cv2.arrowedLine(img, (y,x), (int(y + 1*paf_x[x,y]), int(x + 6*paf_y[x,y])), colors[i], 1, cv2.LINE_AA, tipLength=1)
                    elif type == 'circles':
                        img = cv2.circle(img, (y, x), 1, colors[i], 1)
    cv2.imshow(winTitle, img)
    cv2.waitKey(0)

    return img



def visualizeSkeleton(img, keypoints, winTitle='Skeleton'):
    img = img.copy()
    height = img.shape[0]
    width = img.shape[1]

    blankImage = np.zeros((height, width, 3), np.uint8)

    for person_keypoints in keypoints:

        neck = (int((person_keypoints[5][0] + person_keypoints[6][0])/2) ,  int((person_keypoints[5][1] + person_keypoints[6][1])/2), 2)
        mhip = (int((person_keypoints[11][0] + person_keypoints[12][0])/2) ,  int((person_keypoints[11][1] + person_keypoints[12][1])/2), 2)
        person_keypoints.append(neck)
        person_keypoints.append(mhip)

        limbs = [(0, 1), (0, 2), (1, 3), (2, 4),
                 (5, 7), (7, 9),
                 (6, 8), (8, 10),
                 (5, 6),
                 (11, 12),
                 (11, 13), (13, 15),
                 (12, 14), (14, 16),
                 (0,17), (17,18)]


        image = img

        nose = (person_keypoints[0][0], person_keypoints[0][1])
        lshd = (person_keypoints[5][0], person_keypoints[5][1])
        rshd = (person_keypoints[6][0], person_keypoints[6][1])
        lhip = (person_keypoints[11][0], person_keypoints[11][1])
        rhip = (person_keypoints[12][0], person_keypoints[12][1])

        neck = (int((lshd[0] + rshd[0]) / 2), int((lshd[1] + rshd[1]) / 2))
        mhip = (int((lhip[0] + rhip[0]) / 2), int((lhip[1] + rhip[1]) / 2))

        for limb in limbs:
            if person_keypoints[limb[0]][2] > 0 and person_keypoints[limb[1]][2] > 0:
                cv2.line(image, (person_keypoints[limb[0]][0],person_keypoints[limb[0]][1]), (person_keypoints[limb[1]][0],person_keypoints[limb[1]][1]), color=(0,180,0), thickness=5, lineType=cv2.LINE_AA )
        cv2.line(image, nose, neck, color=(0,180,0), thickness=5, lineType=cv2.LINE_AA )
        cv2.line(image, neck, mhip, color=(0,180,0), thickness=5, lineType=cv2.LINE_AA )

        for kp in person_keypoints:
            cv2.circle(image, (kp[0],kp[1]), radius=5, color=(255,255,255), thickness=-1, lineType=cv2.LINE_AA)

    cv2.imshow(winTitle, img)
    cv2.waitKey(0)

    return image


