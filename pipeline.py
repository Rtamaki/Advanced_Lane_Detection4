import numpy as np
import cv2
import camera_calibration
import image_processing
import lane_detection
import curvature_estimatio
import time



def pipeline1(img, M, Minv):

    undist = image_processing.undistort_image(img)
    binary = 255 *image_processing.compound_thresh(undist,
                                                   sobel_kernel=3,
                                                   abs_sobel_thresh=(120, 255),
                                                   s_thresh=(90, 255),
                    grad_dir_thresh=(0, np.pi/2)).astype(np.uint8)

    binary_warped = image_processing.do_perspective_transform(binary, M)


    cropped_img = image_processing.crop_image(binary_warped, [0, 50], [height, width+170])

    output = lane_detection.draw_lane_estimation(binary_warped,
                                                 window_height=50,
                                                 window_width=40,
                                                 margin=100)
    curvature = curvature_estimatio.get_lanes_curvatures(binary_warped,
                                                         window_width=50,
                                                         window_height=40,
                                                         margin=50,
                                                         extra_img=output)

    return output


# For each image, do all the process of undistortion, threshold application, perspective transformation
# lane points identification, and curvature estimation
def pipeline2(img, M, Minv):

    # Undistort image
    undist = image_processing.undistort_image(img)

    # Apply threshold
    binary = image_processing.compound_thresh(undist,
                                                   sobel_kernel=3,
                                                   abs_sobel_thresh=(120, 255),
                                                   s_thresh=(90, 255),
                    grad_dir_thresh=(0, np.pi/2)).astype(np.uint8)
    # Apply perspective transformation
    binary_warped = image_processing.do_perspective_transform(binary, M)

    # Get lane information
    ret, left_fit, right_fit, leftx, lefty, rightx, righty, nonzerox, nonzeroy, out_img, result = \
        lane_detection.find_polyfit(binary_warped, nwindows=6, margin=80, minpix=50, min_points=50, viz=True)

    # Get estimated lane points from polnomial fit
    ploty, left_fitx, right_fitx = lane_detection.get_estimated_lane_points(binary_warped, left_fit, right_fit)


    image_processing.test_polinomial_fit(out_img, ploty, left_fit)
    image_processing.test_polinomial_fit(out_img, ploty, right_fit)


    # Estimate curvatures from left and right lanes
    left_curvature, right_curvature = \
        curvature_estimatio.calculate_curvatures(leftx, lefty, rightx, righty, ym_per_pix=30.0/780, xm_per_pix=3.7/320)

    # Process undistorted image to visualize where the algorithm think the lane lines are
    drawed_lane_img = \
        image_processing.viz_lane_img(binary_warped, undist, left_fitx, right_fitx, ploty, Minv)

    # Comment on the upper right position of the image the estimated curvature of the lane
    image_processing.annotate_img(drawed_lane_img, str(right_curvature * 0.5 + left_curvature * 0.5), position=(700,50), size=2, color=(0,0,0))
    return drawed_lane_img





def video_pipeline(img):
    offset = 200
    width = 500
    height = 1200
    M, Minv = \
        image_processing.get_perspective_transform_matrix(
            src_corners=np.float32([[600, 450], [995, 650], [325, 650], [685, 450]]),
            dist_corners=np.float32([[offset, offset], [width, height], [offset, height], [width, offset]]))

    result = pipeline2(img, M, Minv)
    return result

#
# offset = 200
# width = 550 #550
# height = 780 #780
# M, Minv = \
#     image_processing.get_perspective_transform_matrix(
#         src_corners=np.float32([[600, 450], [995, 650], [325, 650], [685, 450]]),
#         dist_corners=np.float32([[offset, offset], [width, height], [offset, height], [width, offset]]))
#
#
# img = cv2.imread("./test_images/test2.jpg")
# # warped = image_processing.do_perspective_transform(img, M)
# warped = pipeline2(img, M, Minv)
# cv2.imshow("x", warped)
# cv2.waitKey(5000)
