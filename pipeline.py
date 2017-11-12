import numpy as np
import cv2
import camera_calibration
import image_processing
import lane_detection
import curvature_estimatio
import time



# lane points identification, and curvature estimation
def pipeline_image(img, M, Minv):

    # Undistort image
    undist = image_processing.undistort_image(img)

    # Apply threshold
    binary = image_processing.comb_thresh(undist).astype(np.uint8)
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

    # Get Deviation from the lane center
    transf_left, transf_right, diff = lane_detection.get_deviation_from_center(leftx, rightx, ploty, Minv, out_img,
                                                                               xm_per_pix=3.7 / 310)

    # Comment on the upper right position of the image the estimated curvature of the lane

    # Comment on the upper right position of the image the estimated curvature of the lane
    string1 = 'Curv: ' + '%.2f'%(right_curvature * 0.5 + left_curvature * 0.5) + '(m)'
    string2 = 'Dev: diff: ' + '%.2f'%diff + '(m)'
    image_processing.annotate_img(drawed_lane_img, string1, string2, position=(700, 50), size=2, color=(0, 0, 0))
    return out_img



# lane points identification, and curvature estimation
def pipeline4(img, M, Minv):

    # Undistort image
    undist = image_processing.undistort_image(img)

    # Apply threshold
    binary = image_processing.comb_thresh_for_video(undist).astype(np.uint8)
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

    # Get Deviation from the lane center
    # The conversion from pixels to meters doesn't necessarily have the same scale, since the scales for the warped and
    # the 'original' image are usually different( and in this case it is indeed)
    transf_left, transf_right, diff = lane_detection.get_deviation_from_center(leftx, rightx, ploty, Minv, out_img,
                                                                               xm_per_pix=3.7 / 310)

    # Comment on the upper right position of the image the estimated curvature of the lane
    string1 = 'Curv: ' + '%.2f'%(right_curvature * 0.5 + left_curvature * 0.5) + '(m)'
    string2 = 'Dev: diff' + '%.2f'%diff + '(m)'
    image_processing.annotate_img(drawed_lane_img, string1, string2, position=(700,50), size=2, color=(0,0,0))

    return drawed_lane_img, left_fitx, right_fitx



# lane points identification, and curvature estimation
# and takes into account the previous estimations for the lane lines
# in addition, we define a range for which we consider the new estimation to be valid
def pipeline5(img, M, Minv, prev_estimation, var):

    # Undistort image
    undist = image_processing.undistort_image(img)

    # Apply threshold
    binary = image_processing.comb_thresh_for_video(undist).astype(np.uint8)
    # Apply perspective transformation
    binary_warped = image_processing.do_perspective_transform(binary, M)

    if var > 10:
        left_fit, right_fit, leftx, lefty, rightx, righty, ret = lane_detection.update_lane_fit(binary_warped, prev_estimation[0], prev_estimation[1], margin=15)
    if var <= 10:
        # Get lane information
        ret, left_fit, right_fit, leftx, lefty, rightx, righty, nonzerox, nonzeroy, out_img, result = \
            lane_detection.find_polyfit(binary_warped, nwindows=6, margin=50, minpix=100, min_points=100, viz=True)


    # Get estimated lane points from polnomial fit
    ploty, left_fitx, right_fitx = lane_detection.get_estimated_lane_points(binary_warped, left_fit, right_fit)





    # Estimate curvatures from left and right lanes
    left_curvature, right_curvature = \
        curvature_estimatio.calculate_curvatures(left_fitx, ploty, right_fitx, ploty, ym_per_pix=30.0/780, xm_per_pix=3.7/310)



    # Process undistorted image to visualize where the algorithm think the lane lines are
    drawed_lane_img = \
        image_processing.viz_lane_img(binary_warped, undist, left_fitx, right_fitx, ploty, Minv)

    # Get Deviation from the lane center
    # using the warped image space
    transf_left, transf_right, diff = lane_detection.get_deviation_from_center(left_fitx, right_fitx, ploty, Minv, binary_warped,
                                                                               xm_per_pix=3.7 / 310)

    # Comment on the upper right position of the image the estimated curvature of the lane
    string1 = 'Curv: ' + '%.2f'%(right_curvature * 0.5 + left_curvature * 0.5) + '(m)'
    string2 = 'Dev: diff: ' + '%.2f'%diff + '(m)'
    image_processing.annotate_img(drawed_lane_img, string1, string2, position=(700,50), size=2, color=(0,0,0))

    return drawed_lane_img, left_fit, right_fit

# define 'global' variable to use in the video pipeline to store and use previous estimations for the lane position

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate




@static_vars(counter=0)
def count():
    count.counter += 1
#
# @static_vars(previous_estimations=[])
# @static_vars(measurements=2)
# def prev_estimations(left_fitx, right_fitx, i):
#     if (len(prev_estimations.previous_estimations) < prev_estimations.measurements):
#         prev_estimations.previous_estimations.append((left_fitx, right_fitx))
#     else:
#         prev_estimations.previous_estimations[i % prev_estimations.measurements] = (left_fitx, right_fitx)


@static_vars(previous_estimation=[])
def prev_lane_estimation(leftx, rightx):
    prev_lane_estimation.previous_estimation = [leftx, rightx]

# Function verifies if the new projection for the lane lines doesn't differ too much from the previous one
# def check_new_estimation(left_fitx, right_fitx, previous_estimation, thresh=70):
#     ret = True
#     height = int(len(previous_estimation[0]) / 10)
#     for i in range(height):
#         if np.abs(previous_estimation[0][i] - left_fitx[i]) > thresh:
#             ret = False
#             break
#         if np.abs(previous_estimation[1][i] - right_fitx[i]) > thresh:
#             ret = False
#             break
#
#     if not(ret):
#         left_fitx[: height] = (9 * previous_estimation[0][: height] + left_fitx[: height]) / 10
#         right_fitx[: height] = (9 * previous_estimation[1][: height] + right_fitx[: height]) / 10
#     return left_fitx, right_fitx


# def verify_new_estimation(left_fitx, right_fitx, prev_estimations, var, acceptance=10):
#
#     if var < 20:
#         return True, left_fitx, right_fitx
#     else:
#
#         mean_left, mean_right = mean_previous_values(prev_estimations)
#         ret = True
#         for i in range(len(mean_left)):
#             if np.abs(mean_left[0] - left_fitx[0]) / np.abs(mean_left[0]) > acceptance or \
#             np.abs(mean_right[0] - right_fitx[0]) / np.abs(mean_right[0]) > acceptance:
#                 # np.abs(mean_left[1] - left_fitx[1]) / np.abs(mean_left[1]) > acceptance or \
#                 # np.abs(mean_left[2] - left_fitx[2]) / np.abs(mean_left[2]) > acceptance or \
#
#                 # np.abs(mean_right[1] - right_fitx[1]) / np.abs(mean_right[1]) > acceptance or \
#                 # np.abs(mean_right[2] - right_fitx[2]) / np.abs(mean_right[2]) > acceptance:
#                 ret = False
#                 print(mean_left)
#                 print(left_fitx)
#         if ret:
#             return ret, left_fitx, right_fitx
#         else:
#             return ret, mean_left, mean_right
#
#
# def mean_previous_values(previous_estimations):
#
#     temp_left = [0, 0, 0]
#     temp_right = [0, 0, 0]
#     lenght = len(previous_estimations)
#     for i in range(lenght):
#         temp_left[0] += previous_estimations[i][0][0] / lenght
#         temp_left[1] += previous_estimations[i][0][1] / lenght
#         temp_left[2] += previous_estimations[i][0][2] / lenght
#
#         temp_right[0] += previous_estimations[i][1][0] / lenght
#         temp_right[1] += previous_estimations[i][1][1] / lenght
#         temp_right[2] += previous_estimations[i][1][2] / lenght
#
#
#     return temp_left, temp_right


def video_pipeline(img):


    offset = 200
    width = 500
    height = 1200
    M, Minv = \
        image_processing.get_perspective_transform_matrix(
            src_corners=np.float32([[600, 450], [995, 650], [325, 650], [685, 450]]),
            dist_corners=np.float32([[offset, offset], [width, height], [offset, height], [width, offset]]))

    var = count.counter
    prev_estim = prev_lane_estimation.previous_estimation

    result, left_fit, right_fit = pipeline5(img, M, Minv, prev_estim, var)
    count()
    prev_lane_estimation(left_fit, right_fit)

    return result

