import cv2
import numpy as np
import matplotlib.pyplot as plt
from lanelines.camera import Camera


class Detector:

    def __init__(self):
        pass

    def binarize(self, image, sobel_kernel=5):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        channel = hls[:, :, 2]

        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(channel, orient='x', sobel_kernel=sobel_kernel, thresh=(20, 100))
        grady = self.abs_sobel_thresh(channel, orient='y', sobel_kernel=sobel_kernel, thresh=(20, 100))
        mag_binary = self.mag_thresh(channel, sobel_kernel=sobel_kernel, thresh=(30, 100))
        dir_binary = self.dir_threshold(channel, sobel_kernel=sobel_kernel, thresh=(0.7, 1.3))

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        plt.imshow(combined, cmap='gray')
        plt.show()

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        sobel = cv2.Sobel(img, cv2.CV_64F, 1 if orient == 'x' else 0, 1 if orient == 'y' else 0, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        return self.apply_threshol(scaled_sobel, thresh=thresh)

    def mag_thresh(self, img, sobel_kernel, thresh=(0, 255)):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        return self.apply_threshol(scaled_sobel, thresh=thresh)

    def dir_threshold(self, img, sobel_kernel, thresh=(0, np.pi/2)):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        arctan = np.arctan2(abs_sobely, abs_sobelx)
        return self.apply_threshol(arctan, thresh=thresh)

    def apply_threshol(self, image, thresh=(0, 255)):
        binary_output = np.zeros_like(image)
        binary_output[(image > thresh[0]) & (image < thresh[1])] = 1
        return binary_output

if __name__ == '__main__':
    detector = Detector()
    image = cv2.imread('../test_images/test5.jpg')
    camera = Camera.load()
    detector.binarize(camera.undistort(image))
