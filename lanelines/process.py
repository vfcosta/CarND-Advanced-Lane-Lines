import numpy as np
import cv2
import os.path
from lanelines.camera import Camera
from lanelines.detector import Detector
import matplotlib.pyplot as plt


def calibrate_camera():
    if not os.path.exists("camera.p"):
        camera = Camera()
        camera.calibrate()
        camera.save()
        return camera
    else:
        return Camera.load()


def detect_perspective(camera):
    if camera.perspective_matrix is not None:
        return
    print("detect perspective")
    image = cv2.imread("../test_images/straight_lines1.jpg")
    source_points = [(581, 461), (703, 461), (1064, 692), (245, 692)]
    camera.detect_perspective(image, source_points=np.float32(source_points))
    camera.save()


def undistort_sample_images(camera):
    for image_name, corners in [("calibration1", (9, 5)), ("calibration2", (8, 6)), ("calibration4", (6, 5))]:
        print("undistort", image_name)
        image = cv2.imread("../camera_cal/"+image_name+".jpg")
        chess = Camera.draw_chessboard(image.copy(), corners_x=corners[0], corners_y=corners[1])
        undistorted = camera.undistort(image)
        cv2.imwrite("../output_images/"+image_name+"_chessboard.jpg", chess)
        cv2.imwrite("../output_images/"+image_name+"_undistorted.jpg", undistorted)


def top_down_sample_images(camera):
    for i, image_name in enumerate(["straight_lines1", "test2"]):
        print("warp", image_name)
        image = cv2.imread("../test_images/"+image_name+".jpg")
        top_down = camera.to_top_down(camera.undistort(image))
        if i == 0:
            cv2.polylines(image, np.int32([camera.perspective_source_points]), 1, (0, 0, 255), 5)
            cv2.imwrite("../output_images/" + image_name + "_perspective.jpg", image)
            cv2.polylines(top_down, np.int32([camera.perspective_dest_points]), 1, (0, 0, 255), 5)

        cv2.imwrite("../output_images/"+image_name+"_top_down.jpg", top_down)


def binarize_sample_images(detector):
    for image_name in os.listdir("../test_images/"):
        image_name = image_name.replace('.jpg', '')
        image = cv2.imread('../test_images/'+image_name+'.jpg')
        camera = Camera.load()
        undistorted = camera.undistort(image)
        binarized = detector.binarize(undistorted)
        cv2.imwrite("../output_images/" + image_name + "_undistored.jpg", undistorted)
        cv2.imwrite("../output_images/" + image_name + "_binarized.jpg", binarized*255)


def generate_histogram(detector):
    for i, image_name in enumerate(["straight_lines1", "test2"]):
        print("histogram", image_name)
        image = cv2.imread("../test_images/"+image_name+".jpg")
        histogram = detector.histogram(image)
        plt.clf()
        plt.plot(histogram)
        plt.savefig("../output_images/" + image_name + "_histogram.jpg")


def execute():
    camera = calibrate_camera()
    detector = Detector(camera)
    undistort_sample_images(camera)
    detect_perspective(camera)
    top_down_sample_images(camera)
    binarize_sample_images(detector)
    generate_histogram(detector)


if __name__ == '__main__':
    execute()