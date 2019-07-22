from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

class sub_img():
    def __init__(self, lower, upper, buffer):
        self.lower = lower
        self.upper = upper
        # (self.size_x, self.size_y) = (3300, 1900)
        self.dX = 0
        self.dY = 0
        self.buffer = buffer
        self.pts = deque(maxlen=self.buffer)
        self.direction= ""
        self.counter=0
        


    def print_frame(self):
        print(self.frame.shape,self.frame)

    def get_frame(self, frame):
        self.frame = frame[:,self.lower:self.upper]
   
    def process_one(self):
        greenLower = (29, 86, 6)
        greenUpper = (64, 255, 255)
        
        blurred = cv2.GaussianBlur(self.frame, (11, 11), 0)
        # print(blurred)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # construct a mask for the color "green", then perform
       
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        cv2.imshow('mask',mask)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask 
        
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None


        if len(cnts) > 0:
           
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            #COG
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            
            if radius > 10:
               
                cv2.circle(self.frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(self.frame, center, 5, (0, 0, 255), -1)
               
                self.pts.appendleft(center)

               
        for i in np.arange(1, len(self.pts)):
           
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue

           
            if self.counter >= 10 and i == 1 and self.pts[-10] is not None:
               
                self.dX = self.pts[-10][0] - self.pts[i][0]
                self.dY = self.pts[-10][1] - self.pts[i][1]
                (dirX, dirY) = ("", "")

                
                if np.abs(self.dX) > 20:
                    dirX = "Right" if np.sign(self.dX) == 1 else "Left"

               
                if np.abs(self.dY) > 10:
                    if np.sign(self.dY) == 1:
                        dirY = "Up" 
                    else:
                        dirY="Down"
                        cv2.putText(self.frame, 'Watering Needed', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 3)

                
                if dirX != "" and dirY != "":
                    self.direction = "{}-{}".format(dirY, dirX)
                    

              
                else:
                    self.direction = dirX if dirX != "" else dirY
                    
            thickness = int(np.sqrt(self.buffer / float(i + 1)) * 2.5)
            cv2.line(self.frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

       
        cv2.putText(self.frame, self.direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 3)
        
        cv2.putText(self.frame, "dx: {}, dy: {}".format(self.dX, self.dY),
            (10, self.frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 255), 1)
        self.counter += 1

        return self.frame




