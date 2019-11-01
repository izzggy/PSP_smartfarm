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


#testing
vs = cv2.VideoCapture('testvideo_pot.mp4') 

#realtime video input
#vs = cv2.VideoCapture(0)


# keep looping
while True:
	
	_get, frame = vs.read()
	
	if _get is None:
		break
	out_frame_list = []
	for i in range(num):
		frame_list[i].get_frame(frame)
		out_frame_list.append(frame_list[i].process_one())

		
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

                
