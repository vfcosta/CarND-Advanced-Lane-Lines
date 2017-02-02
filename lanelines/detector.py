import numpy as np
import cv2
from lanelines.binarizer import Binarizer


class Detector:

    def __init__(self, camera):
        self.camera = camera
        self.binarizer = Binarizer()
        self.meters_per_pixels = (3.7 / 700, 30 / 720)  # meters per pixel in x, y

    def detect(self, image, merge_original=False, previous_line_left=None, previous_line_right=None):
        histogram, top_down = self.histogram(image)
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        line_left, out_image_left = self.find_line(top_down, leftx_base, line=previous_line_left)
        line_right, out_image_right = self.find_line(top_down, rightx_base, line=previous_line_right)
        out_image = np.uint8(np.dstack((top_down, top_down, top_down))*255) if merge_original else np.zeros_like(image)
        self.draw_lines(out_image, line_left, out_image_left)
        self.draw_lines(out_image, line_right, out_image_right, channel=2)

        curvature_left = self.calculate_curvature(top_down, line_left)
        curvature_right = self.calculate_curvature(top_down, line_right)
        offset = self.center_offset(image, line_left, line_right)
        return out_image, line_left, line_right, curvature_left, curvature_right, offset

    def center_offset(self, image, line_left, line_right):
        y = image.shape[0]
        linex = line_left[0] * y ** 2 + line_left[1] * y + line_left[2]
        liney = line_right[0] * y ** 2 + line_right[1] * y + line_right[2]
        return (np.mean([linex, liney]) - image.shape[1]/2) * self.meters_per_pixels[0]

    def calculate_curvature(self, image, line):
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        y_eval = np.max(ploty)
        linex = line[0] * ploty ** 2 + line[1] * ploty + line[2]

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * self.meters_per_pixels[1], linex * self.meters_per_pixels[0], 2)
        # Calculate the new radii of curvature
        return ((1 + (2 * left_fit_cr[0] * y_eval * self.meters_per_pixels[1] + left_fit_cr[1]) ** 2) ** 1.5)\
               / np.absolute(2 * left_fit_cr[0])

    def draw_lines(self, image, line, out_image, channel=0):
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        line_fitx = line[0] * ploty ** 2 + line[1] * ploty + line[2]
        image[:, :, channel] = out_image
        cv2.polylines(image, np.int32([np.vstack((line_fitx, ploty)).transpose()]), False, (0, 255, 255), 5)

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

        lanex = nonzerox[lane_inds]
        laney = nonzeroy[lane_inds]

        line = np.polyfit(laney, lanex, 2)
        out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = 255
        return line, out_img

    def search_previous_line(self, image, line, margin, nonzerox, nonzeroy, draw_window):
        out_img = np.zeros_like(image)
        if draw_window:
            ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
            linex = line[0] * ploty ** 2 + line[1] * ploty + line[2]
            line_window1 = np.array([np.transpose(np.vstack([linex - margin, ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([linex + margin, ploty])))])
            line_pts = np.hstack((line_window1, line_window2))
            cv2.fillPoly(out_img, np.int_([line_pts]), 80)

        lane_inds = ((nonzerox > (line[0] * (nonzeroy ** 2) + line[1] * nonzeroy + line[2] - margin)) &
                     (nonzerox < (line[0] * (nonzeroy ** 2) + line[1] * nonzeroy + line[2] + margin)))
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

    def histogram(self, image, bin_width=None):
        binarized = self.binarizer.binarize(self.camera.undistort(image))
        top_down = self.camera.to_top_down(binarized)
        hist = np.mean(top_down[top_down.shape[0] // 2:, :], axis=0)
        if bin_width:
            return np.histogram(range(hist.shape[0]), bins=hist.shape[0]//bin_width, weights=hist)[0], top_down
        return hist, top_down
