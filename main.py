# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 18:19:06 2021

@author: ahmed
"""


import cv2
import mediapipe as mp
import time
import HandTrackingModule as HTM 
import PoseTrackingModule as PTM



def main():
    pTime = 0
    cTime = 0
    handDetector = HTM.handDetector()
    poseDetector = PTM.PoseDetector()
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #captureDevice = camera
    hand_positions = []
    pose_positions = []
    while True:
        success, img = cap.read()
        #overwrite drawn image draw is set true by default
        img = handDetector.find_hands(img)
        lmListHand = handDetector.find_position(img)
        #print(lmListHand)
        hand_positions.append(lmListHand)
        img = poseDetector.find_pose(img)
        lmListPose = poseDetector.find_position(img)
        pose_positions.append(lmListPose)
        
        
        cTime = time.time()
        fps = 1/ (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        
        cv2.imshow("Image",img)
        if(cv2.waitKey(1) == 27):
            cv2.destroyWindow("Image")
            '''
            with open('pose.csv','w') as file:
                file.writelines(["%s\n" % item  for item in lmListPose])
            with open('hands.csv','w') as file:
                file.writelines(["%s\n" % item  for item in hand_positions])
            '''
            break
        


'''
def main():
    pTime = 0
    cTime = 0

    detector = HTM.handDetector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #captureDevice = camera
    while True:
        success, img = cap.read()
        #overwrite drawn image draw is set true by default
        img = detector.find_hands(img)
        lmList = detector.find_position(img)
        print(lmList)
        cTime = time.time()
        fps = 1/ (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        
        cv2.imshow("Image",img)
        if(cv2.waitKey(1) == 27):
            cv2.destroyWindow("Image")
            break
'''
    
if __name__ == "__main__":
    main()
