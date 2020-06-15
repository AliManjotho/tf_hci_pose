class Dataset:

    def __init__(self):
        self._name = ""
        self._nKeypoints = 0
        self._numTrainImages = 0
        self._numTestImages = 0
        self._numValImages = 0


    def __str__(self):
        return "Name: {0}\nNum: of keypoints: {1}\n" \
               "Training Images: {2}\n" \
               "Validation Images: {3}\n" \
               "Test Images: {4}\n".format(self._name, self._nKeypoints, self._numTrainImages, self._numValImages,self._numTestImages)