
## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[undist1]: ./output_images/undist_calibration2.jpg
[undist2]: ./output_images/undist_calibration3.jpg
[undist3]: ./output_images/undist_calibration4.jpg

[bin1]: ./binary_straight_lines1.jpg "Straight Lines Binary 1"
[bin2]: ./binary_straight_lines2.jpg "Straight Lines Binary 2"
[bin3]: ./binary_test1.jpg "Binary Test 1"
[bin4]: ./binary_test2.jpg "Binary Test 2"
[bin5]: ./binary_test3.jpg "Binary Test 3"
[bin6]: ./binary_test4.jpg "Binary Test 4"
[bin7]: ./binary_test5.jpg "Binary Test 5"
[bin8]: ./binary_test6.jpg "Binary Test 6"

[bin_warped1]: ./binary__warped_straight_lines1.jpg "Warped Straight Lines Binary 1"
[bin_warped2]: ./binary__warped_straight_lines2.jpg "Warped Straight Lines Binary 2"
[bin_warped3]: ./binary__warped_test1.jpg "Warped Binary Test 1"
[bin_warped4]: ./binary__warped_test2.jpg "Warped Binary Test 2"
[bin_warped5]: ./binary__warped_test3.jpg "Warped Binary Test 3"
[bin_warped6]: ./binary__warped_test4.jpg "Warped Binary Test 4"
[bin_warped7]: ./binary__warped_test5.jpg "Warped Binary Test 5"
[bin_warped8]: ./binary__warped_test6.jpg "Warped Binary Test 6"

[final_straight1]: ./final_straight_lines1.jpg "Warped Straight Lines Binary 1"
[final_straight2]: ./final_straight_lines2.jpg "Warped Straight Lines Binary 2"
[final_test1]: ./final_test1.jpg "Warped Binary Test 1"
[final_test2]: ./final_test2.jpg "Warped Binary Test 2"
[final_test3]: ./final_test3.jpg "Warped Binary Test 3"
[final_test4]: ./final_test4.jpg "Warped Binary Test 4"
[final_test5]: ./final_test5.jpg "Warped Binary Test 5"
[final_test6]: ./final_test6.jpg "Warped Binary Test 6"



### The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

The Steps
----
To accomplish the objective of this project, we can devide all the process in 2 separable parts: the first is the camera calibration through the images in the folder cam_calibration and the second it the pipeline to get the lane curvature and its position in the image. 

#### The Camera Calibration
----
The camera calibration is the most essential process when working with images, since every camera imposes some kind of distortion which we need to correct to have usefull measurements, and also to estimate the real measurements with the camera matrix (which won't be done in here, but is needed when one work with fiducial markers such as ArUco Markers). To estimate the camera's parameters (distortion coefficients and camera matrix) we use the OpenCV module, more specificaly, the function "cv2.findChessboardCorners()" and "cv2.calibrateCamera()". All the code for camera calibration is located in the fle *camera_calibration.py* in function *calibrateCamera()*.


The steps to get the camera matrix are:
  1) Get the name of all the images in the folder. 
  2) Define the configuration of the chess board and the objective points(Ex: [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]) for each of the images. This is an important step since the wrong order of objective points to the image points given by the function "cv2.findChessboardCorners" will give the wrong camera matrix and distortion. In addition, one can suppose all the images have the same configuration since it is the same chessboard, but this is only true if all the board is always visible, which doesn't happen for the first image.
  3) With the function "cv2.findChessboardCorners" get the corners of the chessboard for each image. In this step it is critical to verify if the corners were found, otherwise we will feed the wrong values to the function "cv2.calibrateCamera". We do this by checking if the *ret* returned by "cv2.findChessboardCorners" is true.
  4) Feed the function "cv2.calibrateCamera" with the objective points and image points. Save the distortion coefficients, the camera matrix and the inverse camera matrix in a .txt file(or any other format) to use it afterwards without having to go through all this process again.

Now we need to check the quality of the camera calibration with the chessboard images. We could do this with any image, but with the board images the distortion is much more obvious. Below are the undistorted images from the folder "camera_cal" "calibration2.jpg", "calibration3.jpg", and "calibration4.jpg".
![][undist1]
![][undist2]
![][undist3]


#### The Lane Detection Pipeline
----
After we have properly found the camera matrix and distortion coefficients, we can go to the next step of finding the lane values, such as the curvature, the deviation from the center, and its position in the image. The process to get these values can be interpreted as a pipeline through which image has to go thorugh. The pipeline used was *pipeline2()* in the folue *pipeline.py*, which calls for several functions in the files *image_processing.py*, *lane_detection_py* and *curavture_estimatio.py*.

  1) Load the camera matrix and distortion coefficients
  2) Undistort the image (*image_processing.undistort_image()*)
  3) Apply a combination of brightness, gradient direction, gradient value, hue thresholds to get a binary image in which the lane lines are as visible as possible(since some loss is acceptable, but they must yet be detectable) and everything else(or at least everything which are near the lane lines in the image), are black. (*image_processing.compound_thresh()*)
    3.1) Transform the image from RGB to HLS color space. This color space is more easily interpretd by use since one dimension is only the "color"(hue), the second is the saturation and the last is the brightness value. So with this color space we already have all the interested dimensions for the algorithm.
    3.2) Apply sobel through "cv2.Sobel" in each axis to get the gradient in each direction, then get the absolute values of the gradient and it's direction. Apply value restriction to both values.
    3.3) Apply value resctricton  in the hue and satuaration dimensions.
    3.4 Combine the the previous binary images through *or* and *and* logic the get the lane lines while excluding everything else as best as possible.
    ![Straight Lines 1 Binary][bin1]
    ![Straight Lines 1 Binary][bin2]
    ![Binary Test 1][bin3]
    ![Binary Test 2][bin4]
    ![Binary Test 3][bin5]
    ![Binary Test 4][bin6]
    ![Binary Test 5][bin7]
    ![Binary Test 6][bin8]

    
  4) From the previous combined binary image, apply perspective transform. (*image_processing.do_perspective_transform()*)
    4.1) To get the correct transformation, we need to use the images *straight_lines1.jpg* and *straight_lines2.jpg* to get the 4 points in the lane which we know define a rectangular form. Afterwards, we need to specify which format these points if they were seen in bird eye view, and we pass the 2 pairs of 4 points to the function "cv2.getPerspectiveTranform" to get the matrix to warp the perspective. 
    4.2) In fact, it is bette to get the perspective transform matrix before, so the pipeline has 1 less step to go through for each image.
    4.3) A point that is interesting to contrast, is the fact that we may want to distort the proportions of the image for the lane detection to work better afterwards.
    4.4) Another important aspect, is to be carefull when choosing the points of the bird eye view, since half of the image may be "upside down".
 
  5) Detect the position of the lane points in the image. We suppose the binary image have as 1 points almost all the lane lines *and* with almost nothing else. This is very important, otherwise the algorithm won't work. (*lane_detection.find_poly_fit()*)
    5.1) Define the number of vertical windows, the width of each window, and the extra margin when the algorithm try to find the lane lines. It's critical to get proper amount of vertical lane lines and the search margin, if there are too many vertical windows, the window won't always find white points(sicen some lane lines are dashed) and if the margin is too big then we can end up detecting white points which aren't related to the lane lines.
    5.2) Get the mean position of the white points in the right and left halves of the image. We suppose these are the lane position in the bottom of the image.
    5.3) If we defined N vertical windows, the algorithm will for (N-1) times, find where there is the highest concentration of white points inside a region defined by the previous estimated position of the lane.
    5.4) Return the pair of 2N
    ![][bin_warped1]  ![][bin_warped2] ![][bin_warped3] ![][bin_warped4]
    ![][bin_warped5]  ![][bin_warped6] ![][bin_warped7] ![][bin_warped8]
    
  6) Estimate a second degree polynomial that fits as perfect as possible in the detected lane points (done together with step 5 in the previous function)
  7) Estimate the lane curvature from the coeficients of this polynomial and a lane point in the image, preferably one close to the car.(*curvature_estimatio.calculate_curvature()*)
  8)  Project the estimated lane from the coeffiecients of the polynomial and apply inverse perspective transform to get the lane position in the original image. Combine the lane image and the original undistorted image with "cv2.addWeighted" (*image_processing.annotate_img()*)
    ![][final_straight1]  ![][final_straight2] ![][final_test1] ![][final_test2]
    ![][final_test3]  ![][final_test4] ![][final_test5] ![][final_test6]


In addition, there are 2 videos for the project. One contains the bird eye view of the binary image of the lane with the estimated polynomial fit for the lane lines and the other is the final video with the estimated curvature and with the lane highlighted in green. This pair is usefull to debug and understand why sometimes the estimation for the lane lines doesn't work as well as they should.


With the video to which the project was tested, we restricted the algorithm to a very restrict situation that may not(and probably doens't) occur always. In the video we had very good light coonditions: no rain, no colored street lights, and no fog. Since our algorithm uses fixed values for selecting the lanes( yellow and white), if during the video we encountered a situation in which the values for the lanes differ significantly, the algortihm won't be able to identify the lane lines. What we could do in such a situation is to use an adaptive thresholding method that adapts to local light values. 
I
n the situation we tested, the lane lines conditions were very satisfactory as well. Even though we had some difficulties regarding the times when the floor was gray, the lane lines were always present. There are situations when the lane lines are partially or compĺeted erased, and then there will be no way to estimate the lane position with the current algorithm.

At last, the training was restricted to curves with very large radius. Since our algorithm defines a fixed margin where we try to identify white points, if the curve happens to be to acute, the algorithm wont't be able to follow.
