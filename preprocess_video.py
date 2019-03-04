#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import tensorflow as tf
import cv2
import utils
import numpy as np
import math
import pickle
import dlib
import face_recognition
import sys
from PIL import Image
#import face_alignment


ECG_SAMPLE_RATE = 16.0
PLE_SAMPLE_RATE = 256.0
N_FRAME = 3600
FRAME_RATE = 30.0
VIDEO_DUR = 120
N_CLIPS = 120
CLIP_SIZE = int(N_FRAME / N_CLIPS)
VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']
LABEL_PATHS = ['D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin']

def Distance(p1,p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx,ny = x,y = center
    sx=sy=1.0
    if new_center:
        (nx,ny) = new_center
    if scale:
        (sx,sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine/sx
    b = sine/sx
    c = x-nx*a-ny*b
    d = -sine/sy
    e = cosine/sy
    f = y-nx*d-ny*e
    rows, cols = image.shape[:2]
    return cv2.warpAffine(image,np.reshape([a,b,c,d,e,f],(2,3)),(rows, cols),flags=cv2.INTER_CUBIC)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
    offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0*offset_h
    # scale factor
    scale = float(dist)/float(reference)
    # rotate original around the left eye
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
    crop_size = (dest_sz[0]*scale, dest_sz[1]*scale*1.3)
    image = image[int(crop_xy[1]):int(crop_xy[1]+crop_size[1]) ,int(crop_xy[0]):int(crop_xy[0]+crop_size[0])]
    # resize it
    image = cv2.resize(image, dest_sz, Image.ANTIALIAS)
    return image


def create_video_clip(video_paths, extra=False):     
    for video_path in video_paths:
        print(video_path)
        print(os.path.exists(video_path))
        path = video_path.split('/')
        if extra:
            if(int(path[4])<10):
                prob_id = 'Proband0'+path[4]
            else:
                prob_id = 'Proband'+path[4]
            cond = '301'   
        else:
            prob_id = path[4]
            cond = path[5].split('_')[0]    
        print(cond)
        print(prob_id)       
        capture = cv2.VideoCapture()
        capture.open(video_path)
        if not capture.isOpened():
            return -1
        else:
            print("video opened. start to read in.....")
        frame_height = int(capture.get(4))
        frame_width = int(capture.get(3))
        nframe = int(capture.get(7))
        print('total frame: '+str(nframe))
        clip = 0
        bboxes = []
        nose_tips = []
        skip_step = 10
        t = math.floor(nframe/skip_step)
#         for idx in range(t):                        
#             print("reading in frame " + str(idx*skip_step)) 
#             capture.set(1, int(idx*skip_step))
#             rd, frame = capture.read()
#             if not rd:
#                 return -1                        
# #             face_locations = face_recognition.face_locations(frame,model='cnn')   
# #             if len(face_locations) != 1:
# #                 print("detecting error!!")
# #                 continue
# #             for (t, r, b, l) in face_locations:          
# #                 bboxes.append([t, r, b, l])
#             face_landmarks = face_recognition.face_landmarks(frame,model="small") 
#             if len(face_landmarks)==0 or len(face_landmarks[0]["nose_tip"])==0:
#                 print("detecting error!!")
#                 continue              
#             nose_tips.append(list(face_landmarks[0]["nose_tip"][0]))
#         if len(nose_tips) == 0:
#             continue
# #         if len(bboxes) == 0:
# #             continue
# #         t, _, _, l = np.min(bboxes, axis=0)
# #         _, r, b, _ = np.max(bboxes, axis=0)
#         nose = np.mean(nose_tips, axis=0)
#        t,r,b,l = 237.3,365,336,265
#         print(str(t)+' '+str(r)+' '+str(b)+' '+str(l))
        capture.set(1, 0)
        for idx in range(nframe):            
            if idx % CLIP_SIZE == 0:
                clip += 1
            if not os.path.exists('./n_processed_video/test/' + cond + '/' + prob_id + '/' + str(clip) + '/'):
                os.makedirs('./n_processed_video/test/' + cond + '/' + prob_id + '/' + str(clip) + '/')
            if idx % 100 == 0:
                print("reading in frame " + str(idx))            
            rd, frame = capture.read()
            if not rd:
                return -1
            if extra:
                lm_li = []
                face_landmarks = face_recognition.face_landmarks(frame,model="large")
                if len(face_landmarks)!=0:
                    for landmark in face_landmarks[0].values():
                        for p in landmark:
                            lm_li.append(list(p))
                    dm_r = np.max(np.array(lm_li),axis=(0))
                    up_l = np.min(np.array(lm_li),axis=(0))
                    nose = (dm_r + up_l ) /2
            y = max(int(nose[1]-140), 0)
            h = min(int(280), (frame_height - y))
            x = max(int(nose[0]-110), 0)
            w = min(int(220), (frame_width - x))
#             y = max(int(0.85*t), 0)
#             h = min(int(1.9*(b-t)), (frame_height - y))
#             x = max(int(0.8*l), 0)
#             w = min(int(1.4*(r-l)), (frame_width - x))
            frame = frame[y:y + h, x:x + w]
            cv2.imwrite( ('./n_processed_video/test/' + cond + '/' + prob_id + '/' + str(clip) + '/' + str(idx) + '.jpg'),frame)             
        print('done:  ' + cond + '/' + prob_id + '/')
        capture.release()


def get_remote_label(label_paths, gt_paths):
    s_dict = {}
    skip_step = 256.0 / 30.0
    gt_skip_step = 16.0 / 30.0
    i = 0
    for label_path, gt_path in zip(label_paths, gt_paths):
        print(i)
        sgns = []
        labels = utils.cvt_sensorSgn(label_path, skip_step)
        gts = utils.cvt_sensorSgn(gt_path, gt_skip_step)
        for idx in range(len(labels) - 1):
            val = float(labels[idx + 1] - labels[idx])
            sgns.append((val,gts[idx]))
        s_dict[str(i)] = sgns
        i += 1
    return s_dict


#if __name__ == '__main__':
        #########1.remote-prepro part videos######################
#     vd, _ = utils.create_file_paths([16],cond='lighting', cond_typ=1)
#     create_video_clip(vd)
    #########2.remote-prepro all videos######################
#     for cond in ['movement']: #, ,'lighting' 'movement'
#         if cond == 'lighting':
#             n = [0,1,3]
#         else:
#             n = [0,1,2]#,1,2
#         for i in n:
#             vd, _ = utils.create_file_paths(range(1, 27), cond=cond, cond_typ=i)
#             create_video_clip(vd)
     #########2-1.remote-prepro extra videos######################
#     vd, _ = utils.create_extra_file_paths(range(15, 18))
#     create_video_clip(vd, extra=True) 
    ############get remote ppg-diff#########################################
    # dict = {}
    # for cond in ['lighting', 'movement']:
    #     if cond == 'lighting':
    #         n = 6
    #     else:
    #         n = 4
    #     for i in range(n):
    # _, lb = utils.create_file_paths(range(1, 27))
    # _, p = utils.create_file_paths(range(1, 27), sensor_sgn=0)
    # s_dict = get_remote_label(lb, p)
    # with open('Pleth.pickle', 'wb') as f:
    #     pickle.dump(s_dict, f)
    # f.close()
