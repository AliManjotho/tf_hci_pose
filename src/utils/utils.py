from math import pow, sqrt, exp
import numpy as np

# GAUSSIAN FUNCTIONS

def gaussian(x, y, x0, y0, sigma, amplitude = 1):
    termX = np.power((x - x0), 2) / (2 * np.power(sigma, 2))
    termY = np.power((y - y0), 2) / (2 * np.power(sigma, 2))
    return amplitude * np.exp(-(termX + termY))





def keypoints2String(keypoints):
    keypointsStr = ''
    for person in keypoints:
        for kp in person:
            keypointsStr = keypointsStr + str(kp[0]) + ',' + str(kp[1]) + ',' + str(kp[2]) + ';'

        keypointsStr = keypointsStr[:-1] + '|'

    keypointsStr = keypointsStr[:-1]

    return keypointsStr


def string2Keypoints(keypointStr):
    keypoints = list()
    personsStr = keypointStr.split('|')

    for personStr in personsStr:
        kpsStr = personStr.split(';')

        personKeypoints = list()
        for kpStr in kpsStr:
            kpPointsStr = kpStr.split(',')

            kp = [int(kpPointsStr[0]), int(kpPointsStr[1]), int(kpPointsStr[2])]
            personKeypoints.append(kp)

        keypoints.append(personKeypoints)

    return np.array(keypoints)