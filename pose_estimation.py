'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


from PIL.Image import fromqimage
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
from skvideo.io import vread, vwrite
import os 


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

    rvecs, tvecs = [], []

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
            # print('rvec', rvec.shape, rvec)
            # print('tvec', tvec.shape, tvec)
            
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners) 

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            rvecs.append(rvec)
            tvecs.append(tvec)  

    return rvecs, tvecs, frame

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Path to video of checkerboard for calibration")
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")

    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    # video = cv2.VideoCapture(0)
    # time.sleep(2.0)

    # while True:
        # ret, frame = video.read()

        # if not ret:
        #     break
        
        # output = pose_estimation(frame, aruco_dict_type, k, d)

        # cv2.imshow('Estimated Pose', output)

        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break

    # video.release()
    # cv2.destroyAllWindows()
    out_dir = 'results'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    video = vread(args['path'])
    # result = []
    out_video = cv2.VideoWriter('with_poses.mp4', 0, 1, (video.shape[2],video.shape[1]))
    for i in range(len(video)):
        print('frame', i)
        frame = video[i]
        rvec, tvec, output = pose_estimation(frame, aruco_dict_type, k, d)
        plt.imsave(out_dir + '/frame' + str(i) + '.jpg', output)
        # result.append(output)
        out_video.write(output)
    
    cv2.destroyAllWindows()
    out_video.release()
    
    # result = np.array(result)
    print('video shape', video.shape)
    # print('result', result.shape)

    # vwrite('with_poses.mp4', result)
    


