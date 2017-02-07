import numpy as np


class Line:

    def __init__(self):
        self.detected = False  # was the line detected in the last iteration?
        self.best_x = None  # average x values of the fitted line over the last n iterations
        self.best_fit = None  # polynomial coefficients averaged over the last n iterations
        self.current_fit = [np.array([False])]  # polynomial coefficients for the most recent fit
        self.radius_of_curvature = None  # radius of curvature of the line in real world units (meters)
        self.line_base_pos = None  # distance in meters of vehicle center from the line
        self.diffs = np.array([0, 0, 0], dtype='float')  # difference in fit coefficients between last and new fits
        self.allx = None  # x values for detected line pixels
        self.ally = None  # y values for detected line pixels
        self.fit_x = None  # x values for last fitted line
        self.fit_y = None  # y values for last fitted line
        self.shape = None  # image shape
        self.meters_per_pixels = (3.7 / 700, 30 / 720)  # meters per pixel in x, y
        self.image = None  # single channel image that contains line pixels

    def fit(self, lanex, laney, shape):
        """Fit a line given valid pixels"""
        self.detected = True
        self.allx = lanex
        self.ally = laney
        self.shape = shape
        self.current_fit = np.polyfit(laney, lanex, 2)
        self.fit_y = np.linspace(0, shape[0] - 1, shape[0])
        self.fit_x = self.current_fit[0] * self.fit_y ** 2 + self.current_fit[1] * self.fit_y + self.current_fit[2]
        self.calculate_curvature()

    def calculate_curvature(self):
        """Calculate the current line curvature"""
        y_eval = self.fit_y[-1]
        # Fit new polynomials to x,y in world space
        fit_curve = np.polyfit(self.fit_y * self.meters_per_pixels[1], self.fit_x * self.meters_per_pixels[0], 2)
        # Calculate the new radii of curvature
        self.radius_of_curvature = ((1 + (2 * fit_curve[0] * y_eval * self.meters_per_pixels[1] + fit_curve[1]) ** 2) ** 1.5)\
               / np.absolute(2 * fit_curve[0])

    def similar_curvature(self, line:  'Line', percent_tolerance=0.25):
        """Verify if the lines have similar curvature with a tolerance of 25%"""
        return abs(self.radius_of_curvature - line.radius_of_curvature) < self.radius_of_curvature * percent_tolerance

    def similar_distance(self, line: 'Line', distance=700, distance_threshold=100):
        """Verify similar distance to another line measured in three points along them"""
        distances = [abs(self.fit_x[int(y)] - line.fit_x[int(y)]) for y in np.linspace(0, self.fit_x.shape[0]-1, 3)]
        return np.all(np.abs(distance - np.array(distances)) < distance_threshold)

    def similar_slope(self, line: 'Line', slope_limit=2, slope_threshold=0.1):
        """Return true for slopes (in f(y)) within slope_limit difference or below a threshold"""
        slope = (self.fit_x[-1] - self.fit_x[0]) / (self.fit_y[-1] - self.fit_y[0])
        slope_other = (line.fit_x[-1] - line.fit_x[0]) / (line.fit_y[-1] - line.fit_y[0])
        return abs(slope - slope_other) < slope_limit or \
                  (abs(slope) > slope_limit and abs(slope_other) < slope_threshold)

    def is_parallel(self, line: 'Line'):
        """
        Combine similar_slope, similar_curvature and similar distance to verify if a line is roughly parallel
        """
        return (self.similar_slope(line) or self.similar_curvature(line)) and self.similar_distance(line)

    def center_offset(self, line):
        return (np.mean([self.fit_x[-1], line.fit_x[-1]]) - self.shape[1] / 2) * self.meters_per_pixels[0]
