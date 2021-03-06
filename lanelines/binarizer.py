import cv2
import numpy as np


class Binarizer:
    """Binarize an image using a combination of strategies."""

    def binarize(self, image, sobel_kernel=5):
        """Binarize image."""
        combined = np.zeros(image.shape[:2])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)  # convert to HLS color space
        s_channel = hls[:, :, 2]  # extract the s channel

        for channel in [gray, s_channel]:
            # Apply each of the thresholding functions
            gradx = self.abs_sobel_thresh(channel, orient='x', sobel_kernel=sobel_kernel, thresh=(80, 150))
            mag_binary = self.mag_thresh(channel, sobel_kernel=sobel_kernel, thresh=(40, 255))
            dir_binary = self.dir_threshold(channel, sobel_kernel=sobel_kernel, thresh=(0.7, 1.3))
            combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        s_binary = self.apply_threshold(s_channel, thresh=(170, 255))
        combined[(s_binary == 1)] = 1
        return combined

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """Apply sobel with threshold."""
        sobel = cv2.Sobel(img, cv2.CV_64F, 1 if orient == 'x' else 0, 1 if orient == 'y' else 0, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        return self.apply_threshold(scaled_sobel, thresh=thresh)

    def mag_thresh(self, img, sobel_kernel, thresh=(0, 255)):
        """Apply sobel in x and y using threshold in magnitude result."""
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        return self.apply_threshold(scaled_sobel, thresh=thresh)

    def dir_threshold(self, img, sobel_kernel, thresh=(0, np.pi/2)):
        """Apply sobel in x and y using direction threshold."""
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.abs(sobelx)
        abs_sobely = np.abs(sobely)
        arctan = np.arctan2(abs_sobely, abs_sobelx)
        return self.apply_threshold(arctan, thresh=thresh)

    def apply_threshold(self, image, thresh=(0, 255)):
        """Apply a threshold to binarize an image given a threshold pair"""
        binary_output = np.zeros_like(image)
        binary_output[(image >= thresh[0]) & (image <= thresh[1])] = 1
        return binary_output
