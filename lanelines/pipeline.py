import cv2
import numpy as np
from lanelines.binarizer import Binarizer
from lanelines.camera import Camera
from lanelines.detector import Detector


camera = Camera.load()
binarizer = Binarizer()
detector = Detector(camera)


def draw_lines(warped_image, original_image, line_left, line_right):
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([line_left.fit_x, line_left.fit_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([line_right.fit_x, line_right.fit_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = camera.to_perspective(color_warp)
    # Combine the result with the original image
    return cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)


def draw_info(image, line_left, line_right, offset):
    cv2.putText(image, "left curvature: %0.1fm" % line_left.radius_of_curvature, (0, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(image, "right curvature: %0.1fm" % line_right.radius_of_curvature, (0, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(image, "center offset: %0.3fm" % offset, (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (255, 255, 255), 3)


def draw_top_down(image, top_down, lines_image, w=320, h=180):
    top_down_resized = cv2.resize(top_down, (w, h))
    lines_resized = cv2.resize(lines_image, (w, h))
    top_down_color = np.uint8(np.dstack((top_down_resized, top_down_resized, top_down_resized)) * 255)
    image[0:h, image.shape[1]-w:] = cv2.addWeighted(top_down_color, 0.5, lines_resized, 1, 0)


def process_image(image, display_top_down=True):
    undistorted = camera.undistort(image)
    binarized = binarizer.binarize(undistorted)
    top_down = camera.to_top_down(binarized)
    lines_image, line_left, line_right, offset = detector.detect(image, top_down)

    lines_original = draw_lines(top_down, image, line_left, line_right)
    draw_info(lines_original, line_left, line_right, offset)
    if display_top_down:
        draw_top_down(lines_original, top_down, lines_image)

    return lines_image, line_left, line_right, offset, lines_original, top_down

