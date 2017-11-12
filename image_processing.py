import cv2
import numpy as np
import camera_calibration

def undistort_image(img):
    mtx, dist, ret = camera_calibration.calibrate_Camera("./camera_cal/")
    if(ret):
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist
    else:
        print ("Couldnt acquire camera matrix and distortion coefficients")
        return img

def draw_corners(img, corners):

    cv2.circle(img, (corners[0]), 5, (0, 255, 0), -1)
    cv2.circle(img, (corners[1]), 5, (0, 255, 0), -1)
    cv2.circle(img, (corners[2]), 5, (0, 255, 0), -1)
    cv2.circle(img, (corners[3]), 5, (0, 255, 0), -1)


def perspective_transform(img, src_corners = [[0, 0], [4, 0], [0, 8], [4, 8]]):
    low_bound = 650
    dist_corners = np.float32([[600, 450], [995, low_bound], [325, low_bound], [685, 450]])
    src_corners = np.float32(src_corners)
    M = cv2.getPerspectiveTransform(dist_corners, src_corners)
    warped = cv2.warpPerspective(img, M, img.shape, flags=cv2.INTER_LINEAR)
    return warped


def get_perspective_transform_matrix(src_corners, dist_corners):
    M = cv2.getPerspectiveTransform(src_corners, dist_corners)
    Minv = cv2.getPerspectiveTransform(dist_corners, src_corners)
    return M, Minv

def do_perspective_transform(img, Matrix):
    warped = cv2.warpPerspective(img, Matrix, (img.shape[0], img.shape[1]), flags=cv2.INTER_LINEAR)
    return warped










def reverse_perspective_transform(point):
    #calculate the point from bird view to orginal view
    low_bound = 650
    dist_corners = np.float32([[600, 450], [980, low_bound], [320, low_bound], [680, 450]])
    src_corners = np.float32([[50, 50], [200, 400], [50, 400], [200, 50]])
    M = cv2.getPerspectiveTransform( dist_corners, src_corners)
    point = [point[0], point[1], 1]
    result = np.matmul(M,point)
    result = result/result[2]
    return result.astype('int')



def get_scaled_sobel_x(img,ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (1+2*ksize,1+2*ksize), 0)
    sobelx = cv2.Sobel(blur_gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel_x = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    return scaled_sobel_x

def get_scaled_sobel_y(img,ksize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (1+2*ksize,1+2*ksize), 0)
    sobely = cv2.Sobel(blur_gray, cv2.CV_64F, 0, 1)
    abs_sobely = np.absolute(sobely)
    scaled_sobel_y = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    return scaled_sobel_y





def threshold_sobel_y(img, thresh=[20,100]):
    scaled_sobel = get_scaled_sobel_y(img)
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def threshold_sobel(gray, thresh=[20,100]):
    abs_sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    abs_sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    gradmag = np.sqrt(abs_sobel_x ** 2 + abs_sobel_y ** 2)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * gradmag / np.max(gradmag))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output



def thresh_grad_dir(gray,sobel_kernel=3, thresh=[0,np.pi/2]):
    abs_sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobel_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))
    sobel_dir = np.arctan2(np.absolute(abs_sobel_y), np.absolute(abs_sobel_x))
    binary_output = np.zeros_like(gray)
    binary_output[(sobel_dir >= thresh[0]) & (sobel_dir <= thresh[1])] = 1
    return binary_output




def compound_thresh2(img, sx_thresh=[0, 255], s_thresh=[0, 255]):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    ### Threshold Sobel
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    abs_sobel_x = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
    abs_sobel_y = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 0, 1))
    gradmag = np.sqrt(abs_sobel_x ** 2 + abs_sobel_y ** 2)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * gradmag / np.max(gradmag))
    # Create a copy and apply the threshold
    sxbinary = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1


    ### Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    color_binary = np.zeros_like(sxbinary)
    color_binary[ (sxbinary == 1) | (s_binary == 1)] = 1
    return color_binary





def hls_select(s_channel, s_thresh=(0, 255)):
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    return s_binary



def compound_thresh(img, sobel_kernel=3, abs_sobel_thresh=(20,100), s_thresh=(120, 255), grad_dir_thresh=(0,np.pi/2)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]
    l_channel = blur_gray = cv2.GaussianBlur(l_channel, (1 + 2 * sobel_kernel, 1 + 2 * sobel_kernel), 0)
    s_channel = blur_gray = cv2.GaussianBlur(s_channel, (1 + 2 * sobel_kernel, 1 + 2 * sobel_kernel), 0)

    grad_dir_bin = thresh_grad_dir(l_channel, sobel_kernel, grad_dir_thresh)
    grad_abs_bin = threshold_sobel(l_channel, abs_sobel_thresh)
    s_binary = hls_select(s_channel, s_thresh)
    binary = np.zeros_like(l_channel)
    binary[(grad_dir_bin == 1) & ((grad_abs_bin == 1) | (s_binary == 1))] = 1
    return binary


def select_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([15,60, 60])
    upper = np.array([30,174, 250])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def select_white(image):
    lower = np.array([222,202,202])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)

    return mask

def comb_thresh(image):
    yellow = select_yellow(image)
    white = select_white(image)

    combined_binary = np.zeros_like(yellow)
    combined_binary[(yellow >= 1) | (white >= 1)] = 1

    return combined_binary


def select_yellow2(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([10,60,60])
    upper = np.array([30,174, 250])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def select_white(image):
    lower = np.array([202,202,202])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)

    return mask

def comb_thresh_for_video(image):
    yellow = select_yellow2(image)
    white = select_white(image)

    combined_binary = np.zeros_like(yellow)
    combined_binary[(yellow >= 1) | (white >= 1)] = 1

    return combined_binary



def crop_image(img, left_upper_corner, right_bottom_corner):
    crop_img = img[left_upper_corner[0]:right_bottom_corner[0], left_upper_corner[1]:right_bottom_corner[1]]
    return crop_img


def annotate_img(img, string1, string2, position=(700,50), size=0.1, color=(0,0,0)):
    cv2.putText(img, string1, position, cv2.FONT_HERSHEY_PLAIN, size,color)
    position2 = (position[0], position[1] + 50)
    cv2.putText(img, string2, position2, cv2.FONT_HERSHEY_PLAIN, size, color)


def viz_lane_img(warped, img, left_fitx, right_fitx, ploty, Minv):

    # Create an image to draw the lines
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_wrap = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y into usable formats for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lanes onto the warped image
    cv2.fillPoly(color_wrap, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_wrap, Minv, (img.shape[1], img.shape[0]))

    result = cv2.addWeighted(img, 1, new_warp, 0.3, 0)
    return result


###
###     Test functions
###


def test_perspective_transform(M):
    img = cv2.imread("./test_images/straight_lines2.jpg")
    warped = do_perspective_transform(img, M)
    cv2.imshow("original", warped)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def test_polinomial_fit(img,ploty,fit, color=(255, 0, 255)):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    for i in range(len(ploty)):
        x = fit[0] * (ploty[i] ** 2) + fit[1] * ploty[i] + fit[2]
        cv2.circle(img, (int(x), int(ploty[i])), 2, color)

