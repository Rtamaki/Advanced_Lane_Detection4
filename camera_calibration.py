import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mtimg
import os

def images_list(folder_path):
    # gets all the file(images) names in the given folder
    return os.listdir(folder_path)

def get_objpoints_imgpoints(img, pattern=(9,6)):
    # from an image, tries to identify the board points and then gives the pair of objpoints and imgpoints
    aux = []
    for i in range(pattern[0]):
        for j in range(pattern[1]):
            aux.append(np.array([i, j, 0], "float32"))
    objpoints = np.array(aux)
    if(len(img.shape) == 3):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    ret, imgpoints = cv2.findChessboardCorners(gray, pattern, None)
    return ret, objpoints, imgpoints

# ########################################################################
# def single_camera_calibration(img, pattern=(9,6)):
#     # from an image, tries to identify the board points and then gives the pair of objpoints and imgpoints
#     aux = []
#     for i in range(pattern[0]):
#         for j in range(pattern[1]):
#             aux.append([j, i])
#     objpoints = aux
#     if (len(img.shape) == 3):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img
#
#     ret, imgpoints = cv2.findChessboardCorners(gray, pattern, None)
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array(objpoints, 'float32'), imgpoints, gray.shape[::-1], None, None)
#     return ret, mtx, dist
#

########################################################################
def calibrate_Camera(folder_path):
    # Calculates camera matrix and distortion coefficients if they weren't already
    try:
        mtx = np.loadtxt("camera_matrix.txt")
        dist = np.loadtxt("dist_coeff.txt")
        ret = True
    except:
        print("No camera matrix found. Will calibrate using the images from ", folder_path)
        imgs_path = images_list(folder_path)
        objpoints_list = []
        imgpoints_list = []
        for i in range(len(imgs_path)):
            img = cv2.imread(folder_path + imgs_path[i])

            ret, objpoints, imgpoints = get_objpoints_imgpoints(img)
            if ret and img is not None:
                objpoints_list.append(objpoints)
                imgpoints_list.append(imgpoints)
            if img is None:
                print("Invalid image", folder_path + imgs_path[i])
            if not ret:
                print("Couldnt find board in the image:", imgs_path[i])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # get the gray image only to feed the cv2.calibrateCamera function
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_list, imgpoints_list, gray.shape[::-1], None, None)
        if ret:
            print("Calibration sucessful")
            np.savetxt("camera_matrix.txt", mtx)
            np.savetxt("dist_coeff.txt", dist)
        else:
            print("Calibration failed")

    return mtx, dist, ret


