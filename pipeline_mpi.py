from mpi4py import MPI

import os
import cv2
import mediapipe as mp
import HandTrackingModule as HTM
import numpy as np
import matplotlib as plt
import pandas as pd
import time

DATADIR = "dataset"


def get_paths(DATADIR="dataset"):
    CATEGORIES = ["correct", "wrong_placement", "totally_wrong"]
    video_dirs = {"correct": [], "wrong_placement": [], "totally_wrong": []}
    paths = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for file in os.listdir(path):
            paths.append(path + '\\' + file)
            if(category == "correct"):
                video_dirs['correct'].append(path + '\\' + file)
            elif(category == "wrong_placement"):
                video_dirs['wrong_placement'].append(path + '\\' + file)
            elif(category == "totally_wrong"):
                video_dirs['totally_wrong'].append(path + '\\' + file)

    return paths


def format_string(s):
    #x = s[1]
    #y = s[2]
    #id= s[0]
    #return {'id':id,'x':x,'y':y}
    return str(s).replace(',', '').replace('[', '').replace(']', '')


def mirror_this(image_file, gray_scale=False, with_plot=False):

    image_mirror = np.fliplr(image_file)
    return image_mirror.astype(np.uint8).copy()





def generate_cs_str(id, frame, screen_width, screen_height, coordinates_left, coordinates_right, label):
    """
    generates csv string from the given arguments
    Note:
    To get the exact number of commas in the string u need calculate (no of elements -1)
    Arguments:
    id:id of the video
    frame: frame number
    screen_width: width of the image 
    screen_height: height of the image
    coordinates_left : list of x,y,z normalized coordinates for left hand
    coordinates_right : list of x,y,z normalized coordinates for right hand
    label : label of the movement correct, incorrect ,incorrect movements and notes
    
    Returns:
    cs_string -- comma separated string
    """
    no_commas = 130
    comma_count = 4
    cs_str = f"{id},{frame},{screen_width},{screen_height}"
    if(len(coordinates_left) == 0):
        #fill in 63 commas (3*21)
        for i in range(63):
            cs_str += ','
            comma_count += 1
    else:
        for coordinate in coordinates_left:
            cs_str += f',{coordinate[1]}'
            cs_str += f',{coordinate[2]}'
            cs_str += f',{coordinate[3]}'
            comma_count += 3
    if(len(coordinates_right) == 0):
        #fill 63 commas (3*21)
        for i in range(63):
            cs_str += ','
            comma_count += 1
    else:
        for coordinate in coordinates_right:
            cs_str += f',{coordinate[1]}'
            cs_str += f',{coordinate[2]}'
            cs_str += f',{coordinate[3]}'
            comma_count += 3

    while(comma_count != no_commas):
        cs_str += ','
        comma_count += 1
    cs_str += f',{label}'

    return cs_str


def extract_frames(write, dirName, fileName, id, movement):
    """
    This function splits each video into multiple frames 
    and generates a csv file of the hand and finger positions 
    
    Arguments:
    write -- boolean indicating to overwrite the image
    dirName -- directory to find the videos
    fileName -- Name of the video you want to be extracted
    id -- ith video being processed
    movement -- movement class
    
    output: None
    """
    drawingModule = mp.solutions.drawing_utils
    handsModule = mp.solutions.hands
    capture = cv2.VideoCapture(fileName)
    detector = HTM.handDetector(max_hands=2)
    frameNr = 0
    while (True):
        #overwrite drawn image draw is set true by default
        success, frame = capture.read()
        #stopping condition
        if not success:
            break
        img = frame
        img = mirror_this(frame)
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

        csv_str = generate_cs_str(
            id, frameNr, screen_width, screen_height, coordinates_left, coordinates_right, movement)

        with open("sequential.csv", "a", newline="") as f:
            f.write(f'{csv_str}\n')

        cv2.imwrite(f'{dirName}/frame_{frameNr}.jpg', img)
        frameNr = frameNr+1

    capture.release()
    print(f"DONE : {fileName}")

def assemble_frames(FRAMESDIR,video_dirs):
    """
    Assembles frames from a directory of images

    Arguments:
    videos directory
    """

    count_videos = 0
    try:
        os.mkdir(FRAMESDIR)
    except Exception as e:
        pass

    for movement in video_dirs:
        count_videos_per_movement = 0
        frames_dir_movement = FRAMESDIR + '/' + movement
        try:
            os.mkdir(frames_dir_movement)
        except Exception as e:
            pass
        #print(video_dirs[movement])
        for path in video_dirs[movement]:
            #print(path)
            #increment needs to be in the inner loop
            try:
                #print(frames_dir_movement + '/'+str(count_videos))
                os.mkdir(frames_dir_movement + '/'+str(count_videos))
            except Exception as e:
                pass
            #dir ill save in,path to video

            extract_frames(True, frames_dir_movement + '/'+str(count_videos),
                        video_dirs[movement][count_videos_per_movement], count_videos, movement)
            count_videos += 1
            count_videos_per_movement += 1
            print(count_videos)

def assemble_frames_mpi(FRAMESDIR,video_dirs,parts,rank):
    start = parts[rank][0] 
    end  = parts[rank][1]
    
    try:
        os.mkdir(FRAMESDIR)
    except Exception as e:
        pass
    #i is the video number
    for i in range(start,end):
        path = video_dirs[i]
        movement = video_dirs[i].split('\\')[-2]
        
        #print(path)
        #increment needs to be in the inner loop
        

        frames_dir_movement = FRAMESDIR + '/' + movement
        try:
            os.mkdir(frames_dir_movement)
        except Exception as e:
            pass
        try:
            #print(frames_dir_movement + '/'+str(count_videos))
            os.mkdir(frames_dir_movement + '/'+str(i))
        except Exception as e:
            pass
        #dir ill save in,path to video
        extract_frames(True, frames_dir_movement + '/'+str(i),
                        video_dirs[i], i, movement)
    
        
        


def partition(video_dirs,num_processes):
    total_num_vids = len(video_dirs)
    
    part = int(total_num_vids / num_processes)
    remainder = (total_num_vids - (part * num_processes))
    #if condition is for the final part we add the remainder and the forumla is just a classc pagination
    partitions = {str(i):[part*i,part+(part*i) + remainder if(i == num_processes -1) else part+(part*i)] for i in range(num_processes)}

    return partitions
    



if __name__ == '__main__':
    
    FRAMESDIR = 'dataset_frames3'
    video_dirs = get_paths()

    group = MPI.COMM_WORLD
    size = group.Get_size()
    rank = group.Get_rank()
    
    parts_dict = partition(video_dirs,size)
    print(parts_dict)
    start = MPI.Wtime()
    assemble_frames_mpi(FRAMESDIR,video_dirs,parts_dict,str(rank))
    end = MPI.Wtime()
    
    if(rank == 0):
        mpi_time = end - start
        print('Average time Time: {:.2f} ms'.format(mpi_time*1000))
    

    
    
    



    
    
