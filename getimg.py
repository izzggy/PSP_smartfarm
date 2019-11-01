
import cv2

def getfirstframe():

    cap = cv2.VideoCapture('testvideo_pot.mp4')  #return capture
    cap.set(cv2.CAP_PROP_POS_FRAMES,50)  #frame
    
    #return bool, frame
    rtn,frame = cap.read()  #return True
    
    cv2.imwrite('image1.jpg', frame)
        
    
    return None
