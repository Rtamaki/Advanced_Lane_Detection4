import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mtplimg
import lane_detection



def get_lanes_fits(leftx, lefty, rightx, righty, ym_per_pix = 30.0 / 720.0 , xm_per_pix = 3.7 / 700.0):

    left_fit = np.polyfit(lefty/ym_per_pix, leftx/xm_per_pix, 2)
    right_fit = np.polyfit(righty/ym_per_pix, rightx/xm_per_pix, 2)
    return left_fit, right_fit



def get_lanes_curvatures(image, window_width=60, window_height=90, margin=50, extra_img=None):

    #get lanes coordinates
    leftx, lefty, rightx, righty = lane_detection.get_lanes_coordinates(image, window_width, window_height, margin)

    # draw_estimated_points(image, leftx, lefty)
    # draw_estimated_points(image, rightx, righty)
    #
    # cv2.imshow('fh',image)
    # cv2.waitKey(5000)

    # Fit new polynomials to x,y in world space
    ym_per_pix = 30.0 / 720.0  # meters per pixel in y dimension
    xm_per_pix = 3.70 / 700.0

    # Get polinomial coeff for left and right lines of the lane
    left_fit_cr, right_fit_cr = get_lanes_fits(leftx, lefty, rightx, righty, ym_per_pix, xm_per_pix)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * lefty * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * righty * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])


    # Now our radius of curvature is in meters
    print(left_curverad, 'l', right_curverad, 'r')
    # Example values: 632.1 m    626.2 m

    #mean between right and left
    fit = np.polyfit(lefty /ym_per_pix, (leftx + rightx) / xm_per_pix / 2, 2)
    curverad = ((1 + (2 * fit[0] * (leftx + rightx) / 2 * ym_per_pix + fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * fit[0])
    print(curverad , "average")
    return np.mean(curverad)















def transform_tuples2vector(tuples):
    y = []
    x = []

    for i in range(len(tuples)):
        y.append(tuples[i][0])
        x.append(tuples[i][1])

    y = np.array(y)
    x = np.array(x)
    return y, x


def draw_estimated_points(img, x_values, y_values):
    fit = np.polyfit(y_values, x_values, 2)
    seq = np.arange(0, img.shape[0], 10)
    for i in range(len(seq)):
        x = fit[0] * seq[i] ** 2 + fit[1] * seq[i] + fit[2]
        cv2.circle(img, (int(x), int(seq[i])), 4, (255, 255, 0), thickness=2)



# This function calculates the curvatre of the right and left lane from the coefficients of the second
# order polynomials of the left and right lins of the lane, independently from which algorithm it came from
def calculate_curvatures(leftx, lefty, rightx, righty, ym_per_pix, xm_per_pix):


    left_fit = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # max_y = np.max(lefty)
    # Here I defined the point where I actually wanted to calculate the curvature, since this is
    # where it would be most interesting to know, because it is some meter ahead of the car
    max_y = 700
    left_curvature = ((1 + (2 * left_fit[0] * max_y * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curvature = ((1 + (2 * right_fit[0] * max_y * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    return left_curvature, right_curvature












