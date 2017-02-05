import numpy as np


class Line:

    def __init__(self):
        self.detected = False  # was the line detected in the last iteration?
        self.recent_xfitted = []  # x values of the last n fits of the line
        self.bestx = None  # average x values of the fitted line over the last n iterations
        self.best_fit = None  # polynomial coefficients averaged over the last n iterations
        self.current_fit = [np.array([False])]  # polynomial coefficients for the most recent fit
        self.radius_of_curvature = None  # radius of curvature of the line in some units
        self.line_base_pos = None  # distance in meters of vehicle center from the line
        self.diffs = np.array([0, 0, 0], dtype='float')  # difference in fit coefficients between last and new fits
        self.allx = None  # x values for detected line pixels
        self.ally = None  # y values for detected line pixels
        self.fit_x = None  # x values for fitted line
        self.fit_y = None  # y values for fitted line
        self.shape = None  # image shape
        self.meters_per_pixels = (3.7 / 700, 30 / 720)  # meters per pixel in x, y

    def fit(self, lanex, laney, shape):
        self.detected = True
        self.allx = lanex
        self.ally = laney
        self.shape = shape
        self.current_fit = np.polyfit(laney, lanex, 2)
        self.fit_y = np.linspace(0, shape[0] - 1, shape[0])
        self.fit_x = self.current_fit[0] * self.fit_y ** 2 + self.current_fit[1] * self.fit_y + self.current_fit[2]
        self.calculate_curvature()

    def calculate_curvature(self):
        y_eval = np.max(self.ally)
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.fit_y * self.meters_per_pixels[1], self.fit_x * self.meters_per_pixels[0], 2)
        # Calculate the new radii of curvature
        self.radius_of_curvature = ((1 + (2 * left_fit_cr[0] * y_eval * self.meters_per_pixels[1] + left_fit_cr[1]) ** 2) ** 1.5)\
               / np.absolute(2 * left_fit_cr[0])
