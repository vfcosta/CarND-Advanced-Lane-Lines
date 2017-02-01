import numpy as np
from lanelines.binarizer import Binarizer


class Detector:

    def __init__(self, camera):
        self.camera = camera
        self.binarizer = Binarizer()

    def detect(self, image):
        histogram = self.histogram(image)
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        print(leftx_base, rightx_base)

    def histogram(self, image):
        binarized = self.binarizer.binarize(self.camera.undistort(image))
        top_down = self.camera.to_top_down(binarized)
        return np.mean(top_down[top_down.shape[0] // 2:, :], axis=0)
