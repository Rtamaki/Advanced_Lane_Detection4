import numpy as np
import cv2
import camera_calibration
import image_processing
import lane_detection
import curvature_estimatio
import time
import pipeline
from moviepy.video import VideoClip
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

image = 'calibration15.jpg'
img = cv2.imread("./camera_cal/"+image)
# warped2 = pipeline.pipeline2(img, M, Minv)
warped2 = image_processing.undistort_image(img)
cv2.imwrite("./undist_"+image,warped2)






# cv2.imshow("x", warped)
cv2.imshow("z", warped2)
cv2.waitKey(5000)
