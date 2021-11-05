# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import cv2
import mediapipe as mp
import time




class PoseDetector():
    def __init__(self,mode = False,
                 complexity = 1,
                 smooth_landmarks = True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 detection_confidence = 0.5,
                 trackConfidence = 0.5):
        
        
        self.mode = mode
        self.complexity = 1
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_confidence = detection_confidence
        self.trackConfidence = trackConfidence
        
        
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.complexity,self.smooth_landmarks,self.enable_segmentation,self.smooth_segmentation,self.detection_confidence,self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        
    def find_pose(self,img,draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.multi_hand_landmarks)
        
        if(self.results.pose_landmarks):
            for poseLms in self.results.pose_landmarks.landmark:
                
                if(draw):
                    self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
                    
                    
        
        return img
    
    def find_position(self,img,poseNo = 0,draw = False):
        lmList = []
        if(self.results.pose_landmarks):
            myPose = self.results.pose_landmarks.landmark
            #print(myPose)
            #print(myHand[0])
            
            for id,lm in enumerate(myPose):
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
    
def main():
    pTime = 0
    cTime = 0
    detector = PoseDetector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #captureDevice = camera
    while True:
        success, img = cap.read()
        #overwrite drawn image draw is set true by default
        img = detector.find_pose(img)
        lmList = detector.find_position(img)
        #print(lmList)
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

    