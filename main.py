import numpy as np
import cv2

import camera_calibration
import image_processing
import lane_detection
import curvature_estimatio
import time
import pipeline
import matplotlib.pyplot as plt



#
# offset = 200
# width = 550 #550
# height = 780 #780
# M, Minv = \
#     image_processing.get_perspective_transform_matrix(
#         src_corners=np.float32([[600, 450], [995, 650], [325, 650], [685, 450]]),
#         dist_corners=np.float32([[offset, offset], [width, height], [offset, height], [width, offset]]))

offset = 200
width = 500
height = 1200
M, Minv = \
    image_processing.get_perspective_transform_matrix(
        src_corners=np.float32([[600, 450], [995, 650], [325, 650], [685, 450]]),
        dist_corners=np.float32([[offset, offset], [width, height], [offset, height], [width, offset]]))

image = 'straight_lines2.jpg'
img = cv2.imread("./test_images/"+image)
warped2 = pipeline.pipeline_image(img, M, Minv)

# This points were used to estimate the distance in pixels of the lane lines in the warped space
# Therefore, the lane has 310 pixels (used straight_lines1.png to calibrate
# cv2.circle(warped2, (500,600), 5, (255, 0, 0), 4)
# cv2.circle(warped2, (190,600), 5, (255, 0, 0), 4)

# cv2.imshow("x", img)
cv2.imshow("z", warped2)
cv2.imwrite('binary_warped_'+image, warped2)

cv2.waitKey(2000)
