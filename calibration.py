'''
Sample Usage:-
python calibration.py --path calibration_checkerboard/ --square_size 0.024
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from skvideo.io import vread


def calibrate(path, square_size, width=6, height=9, visualize=False):
    """ Apply camera calibration operation for images in the given directory path. """

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(width,height,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # images = os.listdir(dirpath)

    # for fname in images:
    #     print(os.path.join(dirpath, fname))
    #     img = cv2.imread(os.path.join(dirpath, fname))
        # print(img)
    video = vread(path)
    if video is None:
        print('video is None')
    else:
        print('video is found')
        print(video.shape)
    for i in range(len(video)):
        img = video[i]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print('finding corners in img' + str(i))
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            print("found corners")
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
        else:
            print('did not find corners')

        if visualize:
            plt.imshow(img)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Path to video of checkerboard for calibration")
    ap.add_argument("-w", "--width", type=int, help="Width of checkerboard (default=9)")
    ap.add_argument("-t", "--height", type=int, help="Height of checkerboard (default=6)")
    ap.add_argument("-s", "--square_size", type=float, default=1, help="Length of one edge (in metres)")
    ap.add_argument("-v", "--visualize", type=str, default="False", help="To visualize each checkerboard image")
    args = vars(ap.parse_args())
    
    path = args['path']
    # 2.4 cm == 0.024 m
    # square_size = 0.024
    square_size = args['square_size']

    if args["visualize"].lower() == "true":
        visualize = True
    else:
        visualize = False

    ret, mtx, dist, rvecs, tvecs = calibrate(path, square_size, visualize=visualize)

    print(mtx)
    print(dist)

    np.save("calibration_matrix", mtx)
    np.save("distortion_coefficients", dist)