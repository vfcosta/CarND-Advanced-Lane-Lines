## Advanced Lane Finding Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

###Camera Calibration

The code for this step is contained in the *Camera* class from [camera.py](lanelines/camera.py).

The method `calibrate` ([camera.py#L24](lanelines/camera.py#L24)) uses chessboard images from `camera_cal` folder to detect corners and calculate the distortion attributes using `cv2.calibrateCamera`.

These images have 9 by 6 corners but in some images not all are visible.
So the `detect_image_points` ([camera.py#L111](lanelines/camera.py#L111)) was used to generate pairs of possible corners visible in chessboard image, e.g.: (9, 6), (9, 5), (9, 4), (8, 6), (7, 6).
The first detected pair of corners was stored in `imgpoints` and the real object points stored in `objpoints`, both used in camera calibration. 

Corner detection examples:

<img src="output_images/calibration1_chessboard.jpg" width="250">
<img src="output_images/calibration2_chessboard.jpg" width="250">
<img src="output_images/calibration4_chessboard.jpg" width="250">

The parameters obtained in calibration was used in `undistort` method ([camera.py#L33](lanelines/camera.py#L33)).
This method calls `cv2.undistort()` function to get an undistorted image.

Distortion correction examples:

<img src="output_images/calibration1_undistorted.jpg" width="250">
<img src="output_images/calibration2_undistorted.jpg" width="250">
<img src="output_images/calibration4_undistorted.jpg" width="250">


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

Distorted images:

<img src="test_images/straight_lines1.jpg" width="250">
<img src="test_images/test2.jpg" width="250">
<img src="test_images/test1.jpg" width="250">

Distortion correction examples:

<img src="output_images/straight_lines1_undistored.jpg" width="250">
<img src="output_images/test2_undistored.jpg" width="250">
<img src="output_images/test1_undistored.jpg" width="250">


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

Binarized examples:

<img src="output_images/straight_lines1_binarized.jpg" width="250">
<img src="output_images/straight_lines2_binarized.jpg" width="250">
<img src="output_images/test4_binarized.jpg" width="250">

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Top down examples:

<img src="output_images/straight_lines1_top_down.jpg" width="250">
<img src="output_images/test2_top_down.jpg" width="250">

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

Detected lines examples:

<img src="output_images/straight_lines1_lines_perspective.jpg" width="250">
<img src="output_images/straight_lines2_lines_perspective.jpg" width="250">
<img src="output_images/test4_lines_perspective.jpg" width="250">

<img src="output_images/test3_lines_perspective.jpg" width="250">
<img src="output_images/test5_lines_perspective.jpg" width="250">
<img src="output_images/test6_lines_perspective.jpg" width="250">


---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4?raw=true)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

