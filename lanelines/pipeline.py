import cv2
import numpy as np
from lanelines.binarizer import Binarizer
from lanelines.camera import Camera
from lanelines.detector import Detector


class Pipeline:

    def __init__(self):
        self.camera = Camera.load()
        self.binarizer = Binarizer()
        self.detector = Detector(self.camera)

    def draw_lane(self, warped_image, original_image, line_left, line_right):
        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([line_left.best_x, line_left.fit_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([line_right.best_x, line_right.fit_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.camera.to_perspective(color_warp)
        # Combine the result with the original image
        return cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

    def draw_info(self, image, line_left, line_right):
        offset = line_left.center_offset(line_right)
        cv2.putText(image, "left curvature: %0.1fm" % line_left.radius_of_curvature, (0, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(image, "right curvature: %0.1fm" % line_right.radius_of_curvature, (0, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(image, "center offset: %0.3fm" % offset, (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 255, 255), 3)

    def draw_top_down(self, image, top_down, lines_image, w=320, h=180):
        top_down_resized = cv2.resize(top_down, (w, h))
        lines_resized = cv2.resize(lines_image, (w, h))
        top_down_color = np.uint8(np.dstack((top_down_resized, top_down_resized, top_down_resized)) * 255)
        image[0:h, image.shape[1]-w:] = cv2.addWeighted(top_down_color, 0.5, lines_resized, 1, 0)

    def process_image(self, image, display_top_down=True):
        undistorted = self.camera.undistort(image)
        binarized = self.binarizer.binarize(undistorted)
        top_down = self.camera.to_top_down(binarized)
        line_left, line_right = self.detector.detect(top_down)

        lines_original = self.draw_lane(top_down, image, line_left, line_right)
        lines_image = self.draw_top_down_lines(image, line_left, line_right)
        self.draw_info(lines_original, line_left, line_right)
        if display_top_down:
            self.draw_top_down(lines_original, top_down, lines_image)

        return lines_image, line_left, line_right, lines_original

    def draw_top_down_lines(self, image, line_left, line_right):
        lines_image = np.zeros_like(image)
        for i, line in enumerate([line_left, line_right]):
            lines_image[:, :, i*2] = line.image
            cv2.polylines(lines_image, np.int32([np.vstack((line.best_x, line.fit_y)).transpose()]), False, (0, 255, 255), 5)
        return lines_image
