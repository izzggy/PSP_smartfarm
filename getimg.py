
import cv2

def getfirstframe():

    cap = cv2.VideoCapture('testvideo_pot.mp4')  #返回一个capture对象
    cap.set(cv2.CAP_PROP_POS_FRAMES,50)  #设置要获取的帧号
    rtn,frame = cap.read()  #read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
    
    cv2.imwrite('image1.jpg', frame)
        
    
    return None
