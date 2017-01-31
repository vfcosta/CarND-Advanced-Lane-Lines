import numpy as np
import cv2
import os.path
from lanelines.camera import Camera
from lanelines.detector import Detector


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


def unwarp_sample_images(camera):
    for filename in ["../output_images/straight_lines1_perspective.jpg", "../test_images/test2.jpg"]:
        print("unwarp", filename)
        output_filename = filename.replace("test_images", "output_images").replace(".jpg", "_unwarped.jpg")
        image = cv2.imread(filename)
        cv2.imwrite(output_filename, camera.unwarp(camera.undistort(image)))


def binarize_sample_images(detector):
    for image_name in os.listdir("../test_images/"):
        image_name = image_name.replace('.jpg', '')
        image = cv2.imread('../test_images/'+image_name+'.jpg')
        camera = Camera.load()
        undistorted = camera.undistort(image)
        binarized = detector.binarize(undistorted)
        cv2.imwrite("../output_images/" + image_name + "_undistored.jpg", undistorted)
        cv2.imwrite("../output_images/" + image_name + "_binarized.jpg", binarized*255)


def execute():
    camera = calibrate_camera()
    undistort_sample_images(camera)
    detect_perspective(camera)
    unwarp_sample_images(camera)
    detector = Detector()
    binarize_sample_images(detector)


if __name__ == '__main__':
    execute()