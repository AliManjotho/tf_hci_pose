import numpy as np


class DataInstance:

    def __init__(self):
        self.__image = 0
        self.__image_file_name = 0
        self.__image_path = 0
        self.__image_width = 0
        self.__image_height = 0
        self.__keypoints = 0
        self.__heatmaps = 0
        self.__pafs = 0
