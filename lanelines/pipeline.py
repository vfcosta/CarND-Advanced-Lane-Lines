from lanelines.binarizer import Binarizer
from lanelines.camera import Camera
from lanelines.detector import Detector


camera = Camera.load()
binarizer = Binarizer()
detector = Detector(camera)


def process_image(image):
    undistorted = camera.undistort(image)
    binarized = binarizer.binarize(undistorted)
    top_down = camera.to_top_down(binarized)
    lines_image, line_left, line_right, offset, lines_original = detector.detect(image, top_down)
    return lines_image, line_left, line_right, offset, lines_original, top_down

