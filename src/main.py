import numpy as np
import cv2
from src.dataset.cocodataset import COCODataset


coco17 = COCODataset()
coco17.loadTrain()
trainDataInstances = coco17.getTrainInstances()
instance = trainDataInstances[561]

img = instance.getImage()
keypoints = instance.getKeypoints()

joints = {  0 : "Nose",
            1 : "LeftEye",
            2 : "RightEye",
            3 : "LeftEar",
            4 : "RightEar",
            5 : "LeftShoulder",
            6 : "RightShoulder",
            7 : "LeftElbow",
            8 : "RightElbow",
            9 : "LeftWrist",
            10 : "RightWrist",
            11 : "LeftHip",
            12 : "RightHip",
            13 : "LeftKnee",
            14 : "RightKnee",
            15 : "LeftAnkle",
            16 : "RightAnkle"  }

LIMBS = [ (0,1),   # Nose - LeftEye
          (0,2),   # Nose - RightEye
          (1,3),   # LeftEye - LeftEar
          (2,4),   # RightEye - RightEar
          (5,7),   # LeftShoulder - LeftElbow
          (7,9),   # LeftElbow - LeftWrist
          (6,8),   # RightShoulder - RightElbow
          (8,10),  # RightElbow - RightWrist
          (11,13), # LeftHip - LeftKnee
          (13,15), # LeftKnee - LeftAnkle
          (12,14), # RightHip - RightKnee
          (14, 16) # RightKnee - RightAnkle
          ]


def getPAF(image, keypoints, sigma_paf, fixedWidthLimbs=True):

    height = image.shape[0]
    width = image.shape[1]
    nLimbs = len(LIMBS)

    out_pafs = np.zeros((nLimbs, 2, height, width))
    n_person_part = np.zeros((nLimbs, height, width))

    for keypoints_person in keypoints:
        for i in range(nLimbs):
            part = LIMBS[i]
            j1 = keypoints_person[part[0]]
            j2 = keypoints_person[part[1]]

            if j1[2] == 2 and j2[2] == 2:
                part_line_segment = np.array([j2[0], j2[1]]) - np.array([j1[0], j1[1]])
                l = np.linalg.norm(part_line_segment)

                if l>1e-2:
                    sigma = sigma_paf
                    if fixedWidthLimbs:
                        sigma = sigma_paf *  l * 0.025

                    v = part_line_segment/l
                    v_per = v[1], -v[0]

                    x, y = np.meshgrid(np.arange(width), np.arange(height))
                    dist_along_part = v[0] * (x - j1[0]) + v[1] * (y - j1[1])
                    dist_per_part = np.abs(v_per[0] * (x - j1[0]) + v_per[1] * (y - j1[1]))

                    mask1 = dist_along_part >= 0
                    mask2 = dist_along_part <= l
                    mask3 = dist_per_part <= sigma
                    mask = mask1 & mask2 & mask3

                    out_pafs[i, 0] = out_pafs[i, 0] + mask.astype('float32') * v[0]
                    out_pafs[i, 1] = out_pafs[i, 1] + mask.astype('float32') * v[1]
                    n_person_part[i] += mask.astype('float32')

    n_person_part = n_person_part.reshape(out_pafs.shape[0], 1, height, width)
    out_pafs = out_pafs/(n_person_part + 1e-8)

    return out_pafs

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
                        img = cv2.arrowedLine(img, (y,x), (int(y + 6*paf_x[x,y]), int(x + 6*paf_y[x,y])), colors[i], 1, cv2.LINE_AA)
                    elif type == 'circles':
                        img = cv2.circle(img, (y, x), 1, colors[i], 1)
    cv2.imshow(winTitle, img)
    cv2.waitKey(0)

    return img

colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]

pafs = getPAF(img, keypoints, 7)
img = visualizePAF(img, pafs, showLimb=-1, winTitle='PAFs', type='circles')
#cv2.imwrite('paf4.jpg', img)

