import numpy as np
import cv2
from lanelines.binarizer import Binarizer
from lanelines.line import Line


class Detector:
    """Detect lane lines in image.

    Uses Camera to apply distortion correction and perspective transformations.
    Uses Binarizer to get a binarized image.
    Uses Line to represent left and right lane lines.
    """

    def __init__(self, camera):
        self.camera = camera
        self.binarizer = Binarizer()
        self.line_left = Line()
        self.line_right = Line()

    def detect(self, top_down):
        """Detect lane lines given an image and its top down projection"""
        leftx_base = None
        rightx_base = None
        if not self.line_left.detected and not self.line_right.detected:
            histogram = self.histogram(top_down)
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        self.line_left = self.find_line(top_down, x=leftx_base, line=self.line_left)
        self.line_right = self.find_line(top_down, x=rightx_base, line=self.line_right)
        parallel = self.line_left.is_parallel(self.line_right)
        self.line_left.validate(parallel)
        self.line_right.validate(parallel)
        return self.line_left, self.line_right

    def find_line(self, image, x, draw_window=True, line=None):
        """Find lane line given a binarized warped image and a start position"""
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if line.detected:
            lane_inds, out_img = self.search_previous_line(image, line, margin=80, nonzerox=nonzerox,
                                                           nonzeroy=nonzeroy, draw_window=draw_window)
        else:
            lane_inds, out_img = self.sliding_window(image, x, margin=80, min_pixels=50, nonzerox=nonzerox,
                                                     nonzeroy=nonzeroy, nwindows=7, draw_window=draw_window)

        if lane_inds.shape[0] > 0:
            out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = 255
            line.fit(nonzerox[lane_inds], nonzeroy[lane_inds], out_img)
        return line

    def search_previous_line(self, image, line, margin, nonzerox, nonzeroy, draw_window):
        """Search current line using previous detected line"""
        out_img = np.zeros_like(image)
        if draw_window:
            line_window1 = np.array([np.transpose(np.vstack([line.best_x - margin, line.fit_y]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([line.best_x + margin, line.fit_y])))])
            line_pts = np.hstack((line_window1, line_window2))
            cv2.fillPoly(out_img, np.int_([line_pts]), 80)

        lane_inds = ((nonzerox > (line.current_fit[0] * (nonzeroy ** 2) + line.current_fit[1] * nonzeroy + line.current_fit[2] - margin)) &
                     (nonzerox < (line.current_fit[0] * (nonzeroy ** 2) + line.current_fit[1] * nonzeroy + line.current_fit[2] + margin)))
        return lane_inds, out_img

    def sliding_window(self, image, x, margin, min_pixels, nonzerox, nonzeroy, nwindows, draw_window):
        """Search current line using sliding window method"""
        out_img = np.zeros_like(image)
        window_height = np.int(image.shape[0] / nwindows)
        lane_inds = []
        for window in range(nwindows):
            left_top = (x - margin, image.shape[0] - window * window_height)
            right_bottom = (x + margin, image.shape[0] - (window + 1) * window_height)
            if draw_window:
                cv2.rectangle(out_img, left_top, right_bottom, 255, 5)
            good_inds = self.nonzero_in_window(left_top, right_bottom, nonzerox, nonzeroy)
            lane_inds.append(good_inds)
            if len(good_inds) > min_pixels:
                x = np.int(np.mean(nonzerox[good_inds]))
        return np.concatenate(lane_inds), out_img

    def nonzero_in_window(self, left_top, right_bottom, nonzerox, nonzeroy):
        """Find nonzero indexes in an image window"""
        return ((nonzeroy >= right_bottom[1]) & (nonzeroy < left_top[1]) &
                (nonzerox >= left_top[0]) & (nonzerox < right_bottom[0])).nonzero()[0]

    def histogram(self, top_down_image, bin_width=None):
        """Calculate image histogram for the bottom half of an image"""
        hist = np.mean(top_down_image[top_down_image.shape[0] // 2:, :], axis=0)
        if bin_width:
            return np.histogram(range(hist.shape[0]), bins=hist.shape[0]//bin_width, weights=hist)[0], top_down_image
        return hist
