from imageai.Prediction.Custom import CustomImagePrediction
import os
import cv2
#import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
execution_path = os.getcwd()

def scanOnce(i,j):
    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(os.path.join(execution_path, "model_ex-011_acc-0.926923.h5"))
    prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
    prediction.loadModel(num_objects=2)

    predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "scanned/scan_" + str(i) + "_" + str(j) + ".jpg"), result_count=2)

    for eachPrediction, eachProbability in zip(predictions, probabilities):
        #print(eachPrediction + " : " + eachProbability)
        if eachPrediction == "pot":
            return float(eachProbability)


detectObject = "training_1_0.jpg"
dtImg = cv2.imread(detectObject,1)
h, w, c = dtImg.shape
#h = 135, w = 75

nameImg = "testImg.jpg"
bgImg = cv2.imread(nameImg,1)
bg_h, bg_w, bg_c = bgImg.shape
#print(bg_w) h = 600, w = 900

scanStep = 15
successX = []
successY = []
#tempX = []
#tempY = []
#Documents/19_Summer_Semester/project3/data_set/models

for i in range(0, bg_w - w + scanStep, scanStep):
    for j in range(0, bg_h - h + scanStep, scanStep):
        tempImg = "scanned/scan_" + str(i) + "_" + str(j) + ".jpg"
        cropImg = bgImg[j:j+int(h),i:i+int(w)]
        cv2.imwrite(tempImg, cropImg, [cv2.IMWRITE_JPEG_QUALITY, 100])
        prob = scanOnce(i,j)
        if prob < 99.84:
            #print(prob)
            if os.path.exists(tempImg):
                os.remove(tempImg)
            else:
                print("The file does not exist")
        else:
            print(prob)
            successX.append(i)
            successY.append(j)

file = open("coordinates.txt","w")
for i in successX:
    for j in successY:
        file.write(str(i+w/2) + "\t" + str(j+h/2) + "\n")

file.close()
