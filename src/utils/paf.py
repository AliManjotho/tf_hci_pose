import numpy as np
from math import sqrt, pow
import cv2

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


def getPAFs(image, keypoints, limbWidth, fixedWidthLimbs=True):

    height = image.shape[0]
    width = image.shape[1]
    nLimbs = len(LIMBS)

    # Create paf matrices 2 for each limb (for x and y unit vectors)
    pafs = np.zeros((nLimbs, 2, height, width))
    nz_parts = np.zeros((nLimbs, height, width))

    # Iterate through keypoints of each person in image
    for person_keypoints in keypoints:

        # Iterate through all limbs for single person
        for limbIndex in range(nLimbs):
            limb = LIMBS[limbIndex]
            j1 = person_keypoints[limb[0]]
            j2 = person_keypoints[limb[1]]

            x1 = j1[0]
            y1 = j1[1]
            v1 = j1[2]
            x2 = j2[0]
            y2 = j2[1]
            v2 = j2[2]

            # If both joints are labeled and visible
            if v1 == 2 and v2 == 2:

                # Limb segment and limb length
                limb_segment = np.array([x2, y2]) - np.array([x1, y1])
                l = sqrt( ((x2 - x1)**2) + ((y2 - y1)**2) )

                if l > 1e-2:
                    limbWidth = limbWidth
                    if fixedWidthLimbs:
                        limbWidth = limbWidth * l * 0.025

                    v = limb_segment / l
                    _v = v[1], -v[0]

                    # Iterate through each pixel of image
                    for p_y in range(height):
                        for p_x in range(width):
                            # Dot product
                            val1 = ( v[0] * (p_x - x1) )  +  ( v[1] * (p_y - y1) )
                            val2 = np.abs( ( _v[0] * (p_x - x1) )  +  ( -v[1] * (p_y - y1) ) )

                            if val1 >=0 and val1 <= l and val2 <=limbWidth:
                                pafs[limbIndex, 0] = v[0]
                                pafs[limbIndex, 1] = v[1]

                                # Count nonzero parts
                                nz_parts[limbIndex] += 1

    nz_parts = nz_parts.reshape(pafs.shape[0], 1, height, width)
    pafs = pafs / (nz_parts + 1e-8)

    return pafs


def getPAF(image, keypoints, limbWidth, fixedWidthLimbs=True):

    height = image.shape[0]
    width = image.shape[1]
    nLimbs = len(LIMBS)

    # Create paf matrices 2 for each limb (for x and y unit vectors)
    pafs = np.zeros((nLimbs, 2, height, width))
    nz_parts = np.zeros((nLimbs, height, width))

    # Iterate through keypoints of each person in image
    for person_keypoints in keypoints:

        # Iterate through all limbs for single person
        for limbIndex in range(nLimbs):
            limb = LIMBS[limbIndex]
            j1 = person_keypoints[limb[0]]
            j2 = person_keypoints[limb[1]]

            x1 = j1[0]
            y1 = j1[1]
            v1 = j1[2]
            x2 = j2[0]
            y2 = j2[1]
            v2 = j2[2]

            # If both joints are labeled and visible
            if v1 == 2 and v2 == 2:

                # Limb segment and limb length
                limb_segment = np.array([x2, y2]) - np.array([x1, y1])
                l = sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

                if l>1e-2:
                    lw = limbWidth
                    if fixedWidthLimbs:
                        lw = limbWidth *  l * 0.025

                    v = limb_segment/l
                    v_ = v[1], -v[0]

                    p_x, p_y = np.meshgrid(np.arange(width), np.arange(height))
                    val1 = ( v[0] * (p_x - x1) )  +  ( v[1] * (p_y - y1) )
                    val2 = np.abs( ( v_[0] * (p_x - x1) )  +  ( v_[1] * (p_y - y1) ) )

                    threshold1 = val1 >= 0
                    threshold2 = val1 <= l
                    threshold3 = val2 <= lw
                    isPixelOnLimb = threshold1 & threshold2 & threshold3

                    pafs[limbIndex, 0] = pafs[limbIndex, 0] + isPixelOnLimb.astype('float32') * v[0]
                    pafs[limbIndex, 1] = pafs[limbIndex, 1] + isPixelOnLimb.astype('float32') * v[1]

                    # Count non-zero values for part pixels
                    nz_parts[limbIndex] += isPixelOnLimb.astype('float32')

    nz_parts = nz_parts.reshape(pafs.shape[0], 1, height, width)
    pafs = pafs/(nz_parts + 1e-8)

    return pafs