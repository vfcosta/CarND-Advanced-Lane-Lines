import numpy as np
import cv2
import glob
import pickle
import os.path

base_dir = os.path.dirname(__file__)


class Camera:
    """Detect and apply distortion and perspective correction"""

    def __init__(self):
        self.shape = None  # image shape
        self.camera_matrix = None  # output matrix from camera calibration
        self.dist_coeffs = None  # distortion coefficients from camera calibration
        self.rvecs = None  # rotation vectors from camera calibration
        self.tvecs = None  # translation vectors from camera calibration
        self.perspective_matrix = None  # matrix for perspective transformation (convert to top down)
        self.inverse_perspective_matrix = None  # matrix for inverse perspective transformation
        self.perspective_source_points = None  # source points used in perspective detection
        self.perspective_dest_points = None  # destination points used in perspective detection

    def calibrate(self, path=os.path.join(base_dir, "../camera_cal/*.jpg"), corners_x=9, corners_y=6, ignore_missing=False):
        """Calibrate camera based on chessboard corners"""
        objpoints, imgpoints, self.shape = self.detect_image_points(path, corners_x=corners_x, corners_y=corners_y,
                                                                    ignore_missing=ignore_missing)
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs =\
            cv2.calibrateCamera(objpoints, imgpoints, self.shape, None, None)
        if not ret:
            raise Exception("can't calibrate camera with path " + path)

    def undistort(self, image):
        """Unistort an image using previous parameters obtained from calibration"""
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

    def detect_perspective(self, image, source_points, dest_points=None, offset_x=300, offset_y=100):
        """Detect perspective from camera"""
        if dest_points is None:
            dest_points = np.float32([(offset_x, offset_y), (image.shape[1] - offset_x, offset_y),
                                      (image.shape[1] - offset_x, image.shape[0]), (offset_x, image.shape[0])])

        self.perspective_source_points = source_points
        self.perspective_dest_points = dest_points
        self.perspective_matrix = cv2.getPerspectiveTransform(source_points, dest_points)
        self.inverse_perspective_matrix = cv2.getPerspectiveTransform(dest_points, source_points)

    def to_top_down(self, image):
        """Convert a given image to top down using the perspective matrix"""
        return cv2.warpPerspective(image, self.perspective_matrix,
                                   (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def to_perspective(self, top_down_image):
        """Convert an image from top down back to perspective using inverse perspective matrix"""
        return cv2.warpPerspective(top_down_image, self.inverse_perspective_matrix,
                                   (top_down_image.shape[1], top_down_image.shape[0]))


    def save(self, filename=os.path.join(base_dir, "camera.p")):
        """Save parameters from Camera class in file"""
        pickle.dump(vars(self), open(filename, "wb"))

    @classmethod
    def load(cls, filename="camera.p"):
        """Load camera parameters from file"""
        camera = Camera()
        attributes = pickle.load(open(filename, "rb"))
        for k, v in attributes.items():
            setattr(camera, k, v)
        return camera

    @staticmethod
    def draw_chessboard(image, corners_x, corners_y):
        """Draw chessboard corners in an image"""
        corners = Camera.find_chessboard(image, corners_x=corners_x, corners_y=corners_y)
        cv2.drawChessboardCorners(image, (corners_x, corners_y), corners, True)
        return image

    @staticmethod
    def find_chessboard(image, corners_x, corners_y):
        """
        Find chessboard in a given image
        :param image: image in BGR format
        :param corners_x: number of inside corners in x
        :param corners_y: number of inside corners in y
        :return: chessboard corners
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, detected_corners = cv2.findChessboardCorners(gray, (corners_x, corners_y), None)
        return detected_corners if ret else None  # return corners if found

    @staticmethod
    def real_world_points(corners_x, corners_y):
        """Get real world points from corners points"""
        points = np.zeros((corners_x * corners_y, 3), np.float32)
        points[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)
        return points

    @staticmethod
    def get_possible_corners(corners_x, corners_y, stop_delta=4):
        """Return pairs of corners_x, corners_y until stop_delta
        Example: [(9, 6), (9, 5), (9, 4), (8, 6), (7, 6), ...]
        """
        range_x = range(corners_x, corners_x - stop_delta, -1)
        range_y = range(corners_y, corners_y - stop_delta, -1)
        return np.array(np.meshgrid(range_x, range_y)).T.reshape(-1, 2)

    @staticmethod
    def detect_image_points(path, corners_x, corners_y, ignore_missing):
        """Detect image points from chessboard using pairs of possible corners"""
        calibration_images = glob.glob(path)
        objpoints = []
        imgpoints = []
        shape = None
        for filename in calibration_images:
            image = cv2.imread(filename)
            shape = image.shape[0:2]
            for nx, ny in Camera.get_possible_corners(corners_x, corners_y):
                corners = Camera.find_chessboard(image, corners_x=nx, corners_y=ny)
                # If found, draw corners
                if corners is not None:
                    print("found corners for", filename, "with", (nx, ny))
                    objpoints.append(Camera.real_world_points(corners_x=nx, corners_y=ny))
                    imgpoints.append(corners)
                    break
                else:
                    print("corners not found for", filename, "with", (nx, ny))
            if not ignore_missing and corners is None:
                raise Exception("can't find corners for " + filename)
        return objpoints, imgpoints, shape
