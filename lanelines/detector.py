import numpy as np
import cv2
from lanelines.binarizer import Binarizer


class Detector:

    def __init__(self, camera):
        self.camera = camera
        self.binarizer = Binarizer()

    def detect(self, image, merge_original=False):
        histogram, top_down = self.histogram(image)
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        line_left, out_image_left = self.find_line(top_down, leftx_base)
        line_right, out_image_right = self.find_line(top_down, rightx_base)
        out_image = np.uint8(np.dstack((top_down, top_down, top_down))*255) if merge_original else np.zeros_like(image)
        self.draw_lines(out_image, line_left, out_image_left)
        self.draw_lines(out_image, line_right, out_image_right, channel=2)
        return out_image

    def draw_lines(self, image, line, out_image, channel=0):
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        line_fitx = line[0] * ploty ** 2 + line[1] * ploty + line[2]
        image[:, :, channel] = out_image
        cv2.polylines(image, np.int32([np.vstack((line_fitx, ploty)).transpose()]), False, (0, 255, 255), 5)

    def find_line(self, image, x, margin=100, min_pixels=50, nwindows=9, draw_window=True):
        """Find lane line given a binarized warped image and a start position"""
        window_height = np.int(image.shape[0] / nwindows)
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        lane_inds = []
        out_img = np.zeros_like(image)

        for window in range(nwindows):
            left_top = (x - margin, image.shape[0] - window * window_height)
            right_bottom = (x + margin, image.shape[0] - (window + 1) * window_height)
            if draw_window:
                cv2.rectangle(out_img, left_top, right_bottom, 255, 5)

            good_inds = ((nonzeroy >= right_bottom[1]) & (nonzeroy < left_top[1]) &
                         (nonzerox >= left_top[0]) & (nonzerox < right_bottom[0])).nonzero()[0]

            lane_inds.append(good_inds)
            if len(good_inds) > min_pixels:
                x = np.int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)
        lanex = nonzerox[lane_inds]
        laney = nonzeroy[lane_inds]

        line = np.polyfit(laney, lanex, 2)
        out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = 255
        return line, out_img

    def histogram(self, image, bin_width=None):
        binarized = self.binarizer.binarize(self.camera.undistort(image))
        top_down = self.camera.to_top_down(binarized)
        hist = np.mean(top_down[top_down.shape[0] // 2:, :], axis=0)
        if bin_width:
            return np.histogram(range(hist.shape[0]), bins=hist.shape[0]//bin_width, weights=hist)[0], top_down
        return hist, top_down
