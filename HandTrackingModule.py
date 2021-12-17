# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:46:05 2021

@author: ahmed
"""
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import time
import csv


class handDetector():
    def __init__(self,mode = False,max_hands = 1,detection_confidence = 0.5,trackConfidence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        
        self.hands = self.mpHands.Hands(self.mode,self.max_hands,1,self.detection_confidence,self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        
    def find_hands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        
        if(self.results.multi_hand_landmarks):
            for handLms in self.results.multi_hand_landmarks:                
                if(draw):
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    def find_position(self,img,handNo = 0,draw = False):
        lmList = []
        if(self.results.multi_hand_landmarks):
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id,lm in enumerate(myHand.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w) , int(lm.y*h)
                #print(id,cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
                
        return lmList
            
            #id is the landmark
            #4,8,12,16,20
            #if(id == 4):
                #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
            #    print(id,str(cx*lm.z),str(cy*lm.z))
    
    '''
    #print(id,lm)
    h,w,c = img.shape
    cx,cy = int(lm.x*w) , int(lm.y*h)
    #print(id,cx,cy)
    
    #id is the landmark
    #4,8,12,16,20
    if(id == 4):
        #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        print(id,str(cx*lm.z),str(cy*lm.z))
        
    
    if(id == 8):
        #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        #print(id,cx,cy)
    if(id == 12):
        #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        #print(id,cx,cy)
    if(id == 16):
        #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        #print(id,cx,cy)
    if(id == 20):
        #cv2.circle(img,(cx,cy),25,(255,0,255),cv2.FILLED)
        #print(id,cx,cy)
    '''
    
def format_string(s):
    #x = s[1]
    #y = s[2]
    #id= s[0]
    #return {'id':id,'x':x,'y':y}
    return str(s).replace(',','').replace('[','').replace(']','')
    
def main():
    pTime = 0
    cTime = 0
    detector = handDetector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #captureDevice = camera
    while True:
        success, img = cap.read()
        #overwrite drawn image draw is set true by default
        img = detector.find_hands(img)
        lmList = detector.find_position(img)
        
        #print(str(lmList))
        
        row = [format_string(item) for item in lmList]
        print(row)
        #.replace('[','').replace(']','')
        
        if(len(lmList) != 0):
            with open("out.csv", "a", newline="") as f:
                
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
                
         
                

                
        cTime = time.time()
        fps = 1/ (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        
        cv2.imshow("Image",img)
        if(cv2.waitKey(1) == 27):
            cv2.destroyWindow("Image")
            break

    
if __name__ == "__main__":
    main()

    