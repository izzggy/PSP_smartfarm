from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from process import *
import maskrcnnGetbox as ms 
import getimg as gi


buffer = 32

gi.getfirstframe()
position_list = ms.position()
num = np.array(np.array(position_list)).shape[0]

frame_list = []
#out_frame_list = []
for i in range(num):
	upper = position_list[i][3]
	lower = position_list[i][1]
	temp_frame = sub_img(lower, upper, buffer)
	frame_list.append(temp_frame)

# upper1 = position_list[0][3]
# lower1 = position_list[0][1]
# upper2 = position_list[1][3]
# lower2 = position_list[1][1]




vs = cv2.VideoCapture('testvideo_pot.mp4')

 
# frame_1 = sub_img(lower1, upper1, buffer)
# frame_2 = sub_img(lower2, upper2, buffer)

# keep looping
while True:
	
	_get, frame = vs.read()
	
	if _get is None:
		break
	out_frame_list = []
	for i in range(num):
		frame_list[i].get_frame(frame)
		out_frame_list.append(frame_list[i].process_one())

	# frame_1.get_frame(frame)
	# frame_2.get_frame(frame)

	# out_1 = frame_1.process_one()
	# out_2 = frame_2.process_one()
	out_frame = np.concatenate(*[out_frame_list], axis=1)
    
	# show the frame to our screen and increment the frame counter
	cv2.imshow("Frame", out_frame)
	key = cv2.waitKey(100) & 0xFF
	
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
 
# if we are not using a video file, stop the camera video stream

vs.stop()
 

vs.release()
 
# close all windows
cv2.destroyAllWindows()

                