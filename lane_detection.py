import cv2
import numpy as np
import image_processing
import camera_calibration
import matplotlib.pyplot as plt


"""
Method 1: Using convolution in a certain range to identify where most white points are.
 The peaks are considered to be the lane points.
"""

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output

def draw_lane_estimation(warped_imaged, window_width = 30, window_height=90, margin=30):
    window_centroids = find_window_centroids2(warped_imaged, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped_imaged)
        r_points = np.zeros_like(warped_imaged)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped_imaged, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped_imaged, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.dstack((warped_imaged, warped_imaged, warped_imaged)) * 255  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display original road image
    else:
        output = np.array(cv2.merge((warped_imaged, warped_imaged, warped_imaged)), np.uint8)
    return output



def get_lanes_coordinates(image, window_width, window_height, margin):
    # get coordinates (y,x) of the lanes
    window_centroids = find_window_centroids2(image, window_width, window_height, margin)
    left_coord = []
    right_coord = []
    lefty =[]
    leftx =[]
    righty = []
    rightx = []
    for i in range(len(window_centroids)):
        lefty.append(image.shape[0] - window_height * i - int(window_height / 2))
        leftx.append(window_centroids[i][0])

        righty.append(image.shape[0] - window_height*i - int(window_height/2))
        rightx.append(window_centroids[i][1])

    return np.array(leftx), np.array(lefty), np.array(rightx), np.array(righty)



def find_window_centroids2(image, window_width, window_height, margin, min_pix=20):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    if np.max(l_sum) > min_pix:
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    else:
        l_center = int(image.shape[1] / 4)

    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    if np.max(r_sum) > min_pix:
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)
    else:
        r_center = int(image.shape[1] * 3 / 4)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)

        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]/2))

        #only update if it found a window that is not the least value possible
        # if i dont do this, when there are no white points the window will go to the most left possible position
        # and I want it to use the previous value instead
        if(np.sum(conv_signal[l_min_index:l_max_index]) > 2):
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, image.shape[1]/2))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        if (np.sum(conv_signal[r_min_index:r_max_index]) > 2):
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids





def find_polyfit_sliding_window(image, window_width=60, window_height=90, margin=50, extra_img=None):

    #get lanes coordinates
    leftx, lefty, rightx, righty = get_lanes_coordinates(image, window_width, window_height, margin)

    # Fit new polynomials to x,y in world space
    ym_per_pix = 30.0 / 720.0  # meters per pixel in y dimension
    xm_per_pix = 3.70 / 700.0

    # Get polinomial coeff for left and right lines of the lane
    left_fit = np.polyfit(lefty / ym_per_pix, leftx / xm_per_pix, 2)
    right_fit = np.polyfit(righty / ym_per_pix, rightx / xm_per_pix, 2)
    return left_fit, right_fit, leftx, lefty, rightx, righty










"""
Method 2: get the indices of where the lines satisfy the minimum amount of white pixels 
    and set the lane line as the mean value of those positions.
"""



def find_polyfit(binary_warped,
                 nwindows=9,  # Set the number of segmentation in the y dimension
                 margin=100,  # Set the width of the windows +/- margin
                 minpix=50,  # Set minimum number of pixels found to recent window
                 min_points=8,  # Least amount of points to consider the detection succesful
                 viz=False  # Sets if we do the visualization for the polynomial fit
                ):
    # take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]*3/4:,:], axis=0)

    #create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # find the peak of the left and right halves of the histogram
    # which will be the starting point for the left and right lanes
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of the windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base



    # Create empty lists to receive left and right lane pixels indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y(and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current -margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
            (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
            (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox <= win_xright_high)).nonzero()[0]


        # If I find > minpix pixels, I use the new value calculated for the mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)


    # Concatenate the array of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixels positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    # Verify if enough points were found. If not, this means an error
    ret = True
    if len(lefty) < min_points or len(righty) < min_points:
        ret = False

    # Find second order polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    result = None
    if viz:
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * (ploty ** 2) + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * (ploty ** 2) + right_fit[1] * ploty + right_fit[2]
        result = visualize(binary_warped, left_lane_inds, right_lane_inds,
                  nonzerox, nonzeroy,
                  margin,
                  left_fitx, right_fitx,
                  ploty)

    return ret, left_fit, right_fit, leftx, lefty, rightx, righty, nonzerox, nonzeroy, out_img, result








def update_lane_fit(binary_warped, left_fit, right_fit, margin=100):

    nonzero = binary_warped.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])


    left_lane_inds = ((nonzerox >= left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin) &
                      (nonzerox <= left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))

    right_lane_inds = ((nonzerox >= right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin) &
                       (nonzerox <= right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin))

    # Extract again left and right line pixels positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    # Fit a new polynomial
    new_left_fit = np.polyfit(lefty, leftx, 2)
    new_right_fit = np.polyfit(righty, rightx, 2)

    # new_left_fit = (new_left_fit + left_fit) / 2
    # new_right_fit = (new_right_fit + right_fit) / 2

    return new_left_fit, new_right_fit, leftx, lefty, rightx, righty, True




# Create a visualization of the second order polynomia for both lane lines
def visualize(binary_warped, left_lane_inds, right_lane_inds,
              nonzerox, nonzeroy,
              margin,
              left_fitx, right_fitx,
              ploty):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # Color in right and left lane pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # and recast x and y points in usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose((np.vstack([left_fitx + margin, ploty]))))])
    left_lane_points = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_lane_points = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_lane_points]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_lane_points]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result




def get_estimated_lane_points(binary_warped, left_fit, right_fit):
    # Generate x and y values fot plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    # ploty = np.linspace(0, 800 - 1, 800)
    left_fitx = left_fit[0] * (ploty ** 2) + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * (ploty ** 2) + right_fit[1] * ploty + right_fit[2]
    return ploty, left_fitx, right_fitx


# def get_deviation_from_center(leftx, rightx, ploty, Minv, original_img, xm_per_pix=3.7/320):
#     i = 1000
#     lx = leftx[i]
#     rx = rightx[i]
#     y = ploty[i]
#
#     transf_left = get_warped_coord([lx, y], Minv)
#     transf_right = get_warped_coord([rx, y], Minv)
#
#     diff = (original_img.shape[1] / 2 - (transf_left[0] + transf_right[0]) / 2) * xm_per_pix
#     return transf_left, transf_right, diff
#
#
# def get_warped_coord(point, M):
#     x_dist = (M[0,0]*point[0] + M[0,1]* point[1] + M[0,2])/ (M[2,0]*point[0] + M[2,1]*point[1] + M[2,2])
#     y_dist = (M[1,0]*point[0] + M[1,1]* point[1] + M[1,2])/ (M[2,0]*point[0] + M[2,1]*point[1] + M[2,2])
#     return (int(x_dist), int(y_dist))
#


def get_deviation_from_center(leftx, rightx, ploty, Minv, warped_img, xm_per_pix=3.7/310):
    i = 1000
    lx = leftx[i]
    rx = rightx[i]
    y = ploty[i]
    diff = (warped_img.shape[1] / 2 - (lx + rx) / 2) * xm_per_pix

    return lx, rx, diff















