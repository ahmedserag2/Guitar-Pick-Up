


import cv2
import mediapipe as mp
import time
import HandTrackingModule as HTM 
import PoseTrackingModule as PTM

import numpy as np
from tuner.tuner_hps import *


def format_string(s):
    #x = s[1]
    #y = s[2]
    #id= s[0]
    #return {'id':id,'x':x,'y':y}
    return str(s).replace(',','').replace('[','').replace(']','')

def mirror_this(image_file, gray_scale=False, with_plot=False):
    
    image_mirror = np.fliplr(image_file)
    return image_mirror.astype(np.uint8).copy() 


def main():
    
    drawingModule = mp.solutions.drawing_utils
    handsModule = mp.solutions.hands
    #guitar/camera/wrong/
    dirName = 'guitar/iphone'
    fileName = 'iphone.MOV'
    capture = cv2.VideoCapture(fileName)
    detector = HTM.handDetector(max_hands=2)
    poseDetector = PTM.PoseDetector()
    frameNr = 0
     

 
    while (True):
        
        #overwrite drawn image draw is set true by default

        success, frame = capture.read()
        if not success:
            break
        img = frame
        img =  mirror_this(frame)
        img = detector.find_hands(img)
        
        #img = poseDetector.find_pose(img)
        #lmListPose = poseDetector.find_position(img)
        
        lmList0 = detector.find_position(img,handNo = 0)
        lmList1 = detector.find_position(img,handNo = 1)
        
        #print(lmList0)
        print(lmList1)
        #row = [format_string(item) for item in lmList0]
        #row1 = [format_string(item) for item in lmList1]
        left = []
        right = []
        try:
            if(lmList0[0][1] - lmList1[0][1] < 0):
                left = lmList0
                right = lmList1
            else:
                left = lmList1
                right = lmList0
        except Exception:
            pass
        
        print(left)
        
        
        row = [format_string(item) for item in left]
        
        if(len(row) != 0):
            with open(f"{dirName}/out.csv", "a", newline="") as f:
                #f.write('{l},'.format(l = str(lmList[0]))
                f.write('{l},'.format(l=str(row[0].split(' ')[1])))
                f.write('{l},'.format(l=str(row[0].split(' ')[2])))
                f.write('{l},'.format(l=str(row[8].split(' ')[1])))
                f.write('{l},'.format(l=str(row[8].split(' ')[2])))
                f.write('{l},'.format(l=str(row[12].split(' ')[1])))
                f.write('{l},'.format(l=str(row[12].split(' ')[2])))
                f.write('{l},'.format(l=str(row[16].split(' ')[1])))
                f.write('{l},'.format(l=str(row[16].split(' ')[2])))
                f.write('{l},'.format(l=str(row[20].split(' ')[1])))
                f.write('{l}\n'.format(l=str(row[20].split(' ')[2])))
                cv2.imwrite(f'{dirName}/frame_{frameNr}.jpg',img)
                frameNr = frameNr+1
        # print(lmList)
        # results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # if results.multi_hand_landmarks != None:
        #     for handLandmarks in results.multi_hand_landmarks:
        #         drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
 
        
     
    capture.release()
        
    
if __name__ == "__main__":
    main()
    