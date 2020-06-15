from math import pow, sqrt, exp
import numpy as np

# GAUSSIAN FUNCTIONS

def gaussian(x, y, x0, y0, sigmax, sigmay, amplitude = 1):
        termX = pow((x - x0), 2) / (2 * pow(sigmax, 2))
        termY = pow((y - y0), 2) / (2 * pow(sigmay, 2))
        return amplitude * exp(-(termX + termY))


def gaussian_OP(x, y, x0, y0, sigma, amplitude = 1):
        distance = pow((x - x0), 2) + pow((y - y0), 2)
        sqrtDistance = sqrt(distance)
        return amplitude * exp(- sqrtDistance / pow(sigma, 2))


def gauss(x, y, x0, y0, sigma, amplitude = 1):
        termX = np.power((x - x0), 2) / (2 * np.power(sigma, 2))
        termY = np.power((y - y0), 2) / (2 * np.power(sigma, 2))
        return amplitude * np.exp(-(termX + termY))
