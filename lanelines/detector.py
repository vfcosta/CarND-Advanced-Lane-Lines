import numpy as np
import cv2
from lanelines.binarizer import Binarizer
from lanelines.line import Line


class Detector:

    def __init__(self, camera):
        self.camera = camera
        self.binarizer = Binarizer()
        self.previous_line_left = None
        self.previous_line_right = None
        self.miss = 0
        self.miss_limit = 10

    def detect(self, image, top_down):
        leftx_base = None
        rightx_base = None
        if self.previous_line_left is None and self.previous_line_right is None:
            histogram = self.histogram(top_down)
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        line_left = self.find_line(top_down, x=leftx_base, line=self.previous_line_left)
        line_right = self.find_line(top_down, x=rightx_base, line=self.previous_line_right)

        # print("PARALLEL", self.miss, line_left.similar_curvature(line_right), line_left.similar_distance(line_right), line_left.similar_slope(line_right))
        line_left, line_right = self.smooth_lines(line_left, line_right)
        offset = self.center_offset(image, line_left, line_right)
        return line_left, line_right, offset

    def smooth_lines(self, line_left, line_right):
        if line_left.is_parallel(line_right):
            self.previous_line_left = line_left
            self.previous_line_right = line_right
            self.miss = 0
        else:
            self.miss += 1
            if self.previous_line_left is not None and self.previous_line_right is not None:
                line_left = self.previous_line_left
                line_right = self.previous_line_right
            if self.miss > self.miss_limit:
                self.previous_line_right = None
                self.previous_line_left = None
        return line_left, line_right

    def center_offset(self, image, line_left, line_right):
        return (np.mean([line_left.fit_x[-1], line_right.fit_x[-1]]) - image.shape[1]/2) * line_left.meters_per_pixels[0]

    def find_line(self, image, x, nwindows=9, draw_window=True, line=None):
        """Find lane line given a binarized warped image and a start position"""
        window_height = np.int(image.shape[0] / nwindows)
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if line is not None:
            lane_inds, out_img = self.search_previous_line(image, line, margin=100, nonzerox=nonzerox,
                                                           nonzeroy=nonzeroy, draw_window=draw_window)
        else:
            lane_inds, out_img = self.sliding_window(image, x, margin=100, min_pixels=50, nonzerox=nonzerox,
                                                     nonzeroy=nonzeroy, nwindows=nwindows,
                                                     window_height=window_height, draw_window=draw_window)

        line = Line()
        line.fit(nonzerox[lane_inds], nonzeroy[lane_inds], image.shape)
        out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = 255
        line.image = out_img
        return line

    def search_previous_line(self, image, line, margin, nonzerox, nonzeroy, draw_window):
        out_img = np.zeros_like(image)
        if draw_window:
            line_window1 = np.array([np.transpose(np.vstack([line.fit_x - margin, line.fit_y]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([line.fit_x + margin, line.fit_y])))])
            line_pts = np.hstack((line_window1, line_window2))
            cv2.fillPoly(out_img, np.int_([line_pts]), 80)

        lane_inds = ((nonzerox > (line.current_fit[0] * (nonzeroy ** 2) + line.current_fit[1] * nonzeroy + line.current_fit[2] - margin)) &
                     (nonzerox < (line.current_fit[0] * (nonzeroy ** 2) + line.current_fit[1] * nonzeroy + line.current_fit[2] + margin)))
        return lane_inds, out_img

    def sliding_window(self, image, x, margin, min_pixels, nonzerox, nonzeroy, nwindows, window_height, draw_window):
        out_img = np.zeros_like(image)
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
        return ((nonzeroy >= right_bottom[1]) & (nonzeroy < left_top[1]) &
                (nonzerox >= left_top[0]) & (nonzerox < right_bottom[0])).nonzero()[0]

    def histogram(self, top_down_image, bin_width=None):
        hist = np.mean(top_down_image[top_down_image.shape[0] // 2:, :], axis=0)
        if bin_width:
            return np.histogram(range(hist.shape[0]), bins=hist.shape[0]//bin_width, weights=hist)[0], top_down_image
        return hist
