import cv2
import mediapipe as mp
import time
import HandTrackingModule as HTM 
import PoseTrackingModule as PTM
import joblib
import numpy as np
from tuner.tuner_hps import *
import sklearn

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
    dirName = 'guitar/labeling3'
    fileName = 'back_cam_guitar.mp4'
    capture = cv2.VideoCapture(fileName)
    detector = HTM.handDetector(max_hands=2)
    frameNr = 0
    index_model = joblib.load("classifier_index.pkl")
    middle_model = joblib.load("classifier_middle.pkl")
    ring_model = joblib.load("classifier_ring.pkl")
    pinky_model = joblib.load("classifier_pinky.pkl")
    
    #print('The scikit-learn version is {}.'.format(sklearn.__version__))

 
    while (True):
        
        #overwrite drawn image draw is set true by default

        success, frame = capture.read()
        if not success:
            break
        img = frame
        img =  mirror_this(frame)
        img = detector.find_hands(img)
        
        hands_dict = detector.find_position2(img)
        coordinates_left = []
        coordinates_right = []
        
        screen_width = None
        screen_height = None
        #if hands are not detected dictionary is null which triggers an error
        if(hands_dict != None):
            #store left and right hand into separate lists depending on the dictionary
            if(len(hands_dict) == 1):
                screen_width = hands_dict[0]['screen_width']
                screen_height = hands_dict[0]['screen_height']
                if(hands_dict[0]['hand_class'] == 'Right'):
                    coordinates_right = hands_dict[0]['pos']
                if(hands_dict[0]['hand_class'] == 'Left'):
                    coordinates_left = hands_dict[0]['pos']
            if(len(hands_dict) == 2):
                screen_width = hands_dict[0]['screen_width']
                screen_height = hands_dict[0]['screen_height']
                if(hands_dict[0]['hand_class'] == 'Left' and hands_dict[1]['hand_class'] == 'Right'):
                    coordinates_left = hands_dict[0]['pos']
                    coordinates_right = hands_dict[1]['pos']
                if(hands_dict[0]['hand_class'] == 'Right' and hands_dict[1]['hand_class'] == 'Left'):
                    coordinates_right = hands_dict[0]['pos']
                    coordinates_left = hands_dict[1]['pos']
    
        #print(coordinates_left)
        if(len(coordinates_left) != 0):
            index_left_coor = np.array([coordinates_left[5][1],coordinates_left[5][2],coordinates_left[5][3],
                                        coordinates_left[6][1],coordinates_left[6][2],coordinates_left[6][3],
                                        coordinates_left[7][1],coordinates_left[7][2],coordinates_left[7][3],
                                        coordinates_left[8][1],coordinates_left[8][2],coordinates_left[8][3]])
            index_left_coor = index_left_coor.reshape(1,12)
            middle_left_coor = np.array([coordinates_left[9][1],coordinates_left[9][2],coordinates_left[9][3],
                                        coordinates_left[10][1],coordinates_left[10][2],coordinates_left[10][3],
                                        coordinates_left[11][1],coordinates_left[11][2],coordinates_left[11][3],
                                        coordinates_left[12][1],coordinates_left[12][2],coordinates_left[12][3]])
            middle_left_coor = middle_left_coor.reshape(1,12)
            
            ring_left_coor = np.array([coordinates_left[13][1],coordinates_left[13][2],coordinates_left[13][3],
                                        coordinates_left[14][1],coordinates_left[14][2],coordinates_left[14][3],
                                        coordinates_left[15][1],coordinates_left[15][2],coordinates_left[15][3],
                                        coordinates_left[16][1],coordinates_left[16][2],coordinates_left[16][3]])
            ring_left_coor = ring_left_coor.reshape(1,12)
            
            pinky_left_coor = np.array([coordinates_left[17][1],coordinates_left[17][2],coordinates_left[17][3],
                                        coordinates_left[18][1],coordinates_left[18][2],coordinates_left[18][3],
                                        coordinates_left[19][1],coordinates_left[19][2],coordinates_left[19][3],
                                        coordinates_left[20][1],coordinates_left[20][2],coordinates_left[20][3]])
            pinky_left_coor = pinky_left_coor.reshape(1,12)
            
            
            print(index_left_coor)
            index_prediction = index_model.predict(index_left_coor)[0]
            middle_prediction = middle_model.predict(middle_left_coor)[0]
            ring_prediction = ring_model.predict(ring_left_coor)[0]
            pinky_prediction = pinky_model.predict(pinky_left_coor)[0]
            
            cv2.putText(img,f"index {index_prediction}",(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            cv2.putText(img,f"middle {middle_prediction}",(10,110), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            cv2.putText(img,f"ring {ring_prediction}",(10,150), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            cv2.putText(img,f"pinky {pinky_prediction}",(10,190), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
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
    