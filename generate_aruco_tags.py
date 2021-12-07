'''
Sample Command:-
python generate_aruco_tags.py --id 24 --type DICT_5X5_100 -o tags/
'''


import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import ARUCO_DICT
import cv2
import sys
import os 

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output folder to save ArUCo tag")
ap.add_argument("-n", "--number_of_tags", type=int, default=16, help="ID of ArUCo tag to generate")
ap.add_argument("-t", "--type", type=str, default="DICT_4X4_100", help="type of ArUCo tag to generate")
ap.add_argument("-s", "--size", type=int, default=200, help="Size of the ArUCo tag")
args = vars(ap.parse_args())


# Check to see if the dictionary is supported
if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])

if not os.path.exists(args['output']):
	os.mkdir(args['output'])

for i in range(args['number_of_tags']):
	print("Generating ArUCo tag of type '{}' with ID '{}'".format(args["type"], i))
	tag_size = args["size"]
	tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
	cv2.aruco.drawMarker(arucoDict, i, tag_size, tag, 1)

	# Save the tag generated
	tag_name = f'{args["output"]}/{args["type"]}_id_{i}.png'
	tag = tag.reshape(tag_size, tag_size)
	# print(tag.shape)
	plt.imsave(tag_name, tag, cmap='gray')
# cv2.imwrite(tag_name, tag)
# cv2.imshow("ArUCo Tag", tag)
# cv2.waitKey(0)
# cv2.destroyAllWindows()