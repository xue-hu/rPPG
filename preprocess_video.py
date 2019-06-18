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
from Deeplab import DeepLabModel
from scipy import signal
import subprocess as sp
import ffmpeg
import matplotlib.pyplot as plt
from jeanCV import skinDetector 

ECG_SAMPLE_RATE = 16.0
PLE_SAMPLE_RATE = 256.0
N_FRAME = 3600
FRAME_RATE = 30.0
VIDEO_DUR = 120
N_CLIPS = 120
CLIP_SIZE = int(N_FRAME / N_CLIPS)
VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']
LABEL_PATHS = ['D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin']
MODEL_PATH = '/Vitalcam_Dataset/FaceRegionDetection/deeplab/datasets/pascal_voc_seg/crop_512/iter_5000/train_on_trainval_set/export/frozen_inference_graph.pb'
seg_model = DeepLabModel(MODEL_PATH)

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
        rd, frame = capture.read()
        cv2.imwrite( '11.jpg',frame)     
#         for idx in range(nframe):
#             if idx % CLIP_SIZE == 0:
#                 clip += 1
#             if idx % 100 == 0:
#                 print("reading in frame " + str(idx)) 
#             rd, frame = capture.read()
#             if not rd:
#                 return -1 
            
#             if idx == 0 :
# #                 pre_map = None
#                 face_landmarks = face_recognition.face_landmarks(frame,model="small")
#                 while len(face_landmarks)==0 or len(face_landmarks[0]["nose_tip"])==0:
#                     print('try again!')
#                     rd, fra = capture.read()
#                     face_landmarks = face_recognition.face_landmarks(fra,model="small")
#                 capture.set(1,1)
#                 if len(face_landmarks)==0 or len(face_landmarks[0]["nose_tip"])==0:
#                     v_path = video_path[:64]+'/101_natural_lighting/'+video_path[89:]
#                     cap = cv2.VideoCapture()
#                     cap.open(v_path)
#                     rd, fra = cap.read()
#                     face_landmarks = face_recognition.face_landmarks(fra,model="small")
#                     cap.release()
#                 nose = face_landmarks[0]["nose_tip"][0]            
#             y = max(int(nose[1]-280), 0)
#             h = min(int(480), (frame_height - y))
#             x = max(int(nose[0]-280), 0)
#             w = min(int(480), (frame_width - x)) 
#             frame = frame[y:y + h, x:x + w]
            
#             image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(image)
#             _, seg_map = seg_model.run(image) 
#             true_points = np.argwhere(seg_map)
#             if seg_map.any() == 0:
#                 print(str(idx)+': no skin!!')
#                 continue
# #             if np.argwhere(seg_map).size < 0.8*np.argwhere(pre_map).size:
# #                 print(str(idx)+': wrong detect!!') 
# #                 continue
# #                 seg_map = pre_map
# #                 true_points = np.argwhere(pre_map)
# #             pre_map = seg_map
#             top_left = true_points.min(axis=0)
#             bottom_right = true_points.max(axis=0)
#             seg_map = seg_map[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
#             frame = frame[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

#             if not os.path.exists('./n_processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'):
#                 os.makedirs('./n_processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/')
#             cv2.imwrite( ('./n_processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/' + str(idx) + '.jpg'),frame)    
#             if not os.path.exists('./n_processed_video/mask/' + cond + '/' + prob_id + '/' + str(clip) + '/'):
#                 os.makedirs('./n_processed_video/mask/' + cond + '/' + prob_id + '/' + str(clip) + '/')
#             cv2.imwrite( ('./n_processed_video/mask/' + cond + '/' + prob_id + '/' + str(clip) + '/' + str(idx) + '.jpg'),seg_map)
        print('done:  ' + cond + '/' + prob_id + '/')
        capture.release()
            
#########origin nose tip static crop###################################################################################
#         bboxes = []
#         nose_tips = []
#         skip_step = 10
#         t = math.floor(nframe/skip_step)
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
#         print(str(t)+' '+str(r)+' '+str(b)+' '+str(l))

#         capture.set(1, 0)
#         for idx in range(nframe):            
#             if idx % CLIP_SIZE == 0:
#                 clip += 1
#             if not os.path.exists('./n_processed_video/test/' + cond + '/' + prob_id + '/' + str(clip) + '/'):
#                 os.makedirs('./n_processed_video/test/' + cond + '/' + prob_id + '/' + str(clip) + '/')
#             if idx % 100 == 0:
#                 print("reading in frame " + str(idx))            
#             rd, frame = capture.read()
#             if not rd:
#                 return -1
#             if extra:
#                 lm_li = []
#                 face_landmarks = face_recognition.face_landmarks(frame,model="large")
#                 if len(face_landmarks)!=0:
#                     for landmark in face_landmarks[0].values():
#                         for p in landmark:
#                             lm_li.append(list(p))
#                     dm_r = np.max(np.array(lm_li),axis=(0))
#                     up_l = np.min(np.array(lm_li),axis=(0))
#                     nose = (dm_r + up_l ) /2
#             y = max(int(nose[1]-140), 0)
#             h = min(int(280), (frame_height - y))
#             x = max(int(nose[0]-110), 0)
#             w = min(int(220), (frame_width - x))
#             y = max(int(0.85*t), 0)
#             h = min(int(1.9*(b-t)), (frame_height - y))
#             x = max(int(0.8*l), 0)
#             w = min(int(1.4*(r-l)), (frame_width - x))
#             frame = frame[y:y + h, x:x + w]
#             cv2.imwrite( ('./n_processed_video/test/' + cond + '/' + prob_id + '/' + str(clip) + '/' + str(idx) + '.jpg'),frame)             
#         print('done:  ' + cond + '/' + prob_id + '/')
#         capture.release()
############################################################################################################


        
def create_mask(video_paths,extra=False):
    for video_path in video_paths:
        path = video_path.split('/')
        if extra:
            if int(path[4])>9:
                prob_id = 'Proband' + path[4]
            else:
                prob_id = 'Proband0' + path[4]
            cond = '301' 
            n_clips = 600
        else:
            prob_id = path[4]
            cond = path[5].split('_')[0]
            n_clips = 120      
        for clip in range(1, int(n_clips + 1)):
            if not os.path.exists('./processed_mask/' + cond + '/' + prob_id + '/' + str(clip) + '/'):
                os.makedirs('./processed_mask/' + cond + '/' + prob_id + '/' + str(clip) + '/')
            scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
            start_pos = (clip - 1) * CLIP_SIZE
            end_pos = clip * CLIP_SIZE       
            print(cond + '-' + prob_id + '-clip' + str(clip))
            for idx in range(start_pos, end_pos):
                if idx % 100 == 0:
                    print("reading in frame " + str(idx)) 
                pre_path = scr_path + str(idx) + '.jpg'               
                if not os.path.exists(pre_path):
                    continue    
                image = Image.open(pre_path)
                _, seg_map = seg_model.run(image) 
                
                #seg_map = np.asarray(seg_map,dtype=np.uint8)
                cv2.imwrite( ('./processed_mask/' + cond + '/' + prob_id + '/' + str(clip) + '/' + str(idx) + '.jpg'),seg_map)             
        print('done:  ' + cond + '/' + prob_id + '/')
        
        
def create_spectrogram(video_paths,window_size = 120, extra=False):
    for video_path in video_paths:
        video_data = []
        sgram = []
        path = video_path.split('/')
        if extra:
            if int(path[4])>9:
                prob_id = 'Proband' + path[4]
            else:
                prob_id = 'Proband0' + path[4]
            cond = '301' 
            n_clips = 600
        else:
            prob_id = path[4]
            cond = path[5].split('_')[0]
            n_clips = 120      
        for clip in range(100, int(n_clips + 1)):
            if not os.path.exists('./spectrogram/' + cond + '/' + prob_id + '/' + str(clip) + '/'):
                os.makedirs('./spectrogram/' + cond + '/' + prob_id + '/' + str(clip) + '/')
            scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
            start_pos = (clip - 1) * CLIP_SIZE
            end_pos = clip * CLIP_SIZE    
            print(cond + '-' + prob_id + '-clip' + str(clip))
            for idx in range(start_pos, end_pos):
                if idx % 100 == 0:
                    print("reading in frame " + str(idx)) 
                pre_path = scr_path + str(idx) + '.jpg'               
                if not os.path.exists(pre_path):
                    continue  
                frame = cv2.imread(pre_path).astype(np.float32) 
                mask = utils.skin_seg(pre_path,112, 112, resized=False)
                frame = cv2.bitwise_and(frame,frame,mask = mask)
                video_data.append(frame[:,:,1].mean())
                if len(video_data) < window_size:
                    continue  
                video_data = np.asarray(video_data, dtype=np.float32)
                f, t, spectrogram = signal.spectrogram(video_data, 30, nperseg=10, nfft=50, noverlap=5)
#                 plt.pcolormesh(t, f, spectrogram)
#                 plt.ylabel('Frequency [Hz]')
#                 plt.xlabel('Time [sec]')
#                 plt.savefig('gram.png')                
#                 return
                sgram.append(spectrogram)
                video_data = []                   
    return
       

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


if __name__ == '__main__':
        #########1.remote-prepro part videos######################
#     vd, _ = utils.create_file_paths([16],cond='lighting', cond_typ=1)
#     create_video_clip(vd)
#     #########2.remote-prepro all videos######################
#     for cond in [ 'lighting','movement']: #, ,'lighting' 'movement'
#         if cond == 'lighting':
#             n = [0,1]
#         else:
#             n = [2]#,1,2
#         for i in n:
#             vd, _ = utils.create_file_paths(range(1,2), cond=cond, cond_typ=i)#8,10,16,
#             create_video_clip(vd)
#             create_mask(vd)
#             create_spectrogram(vd)
     #######2-1.remote-prepro extra videos######################
#     vd, _ = utils.create_extra_file_paths(range(7, 18))
#     #create_video_clip(vd, extra=True) 
#     create_mask(vd, extra=True) 
#     create_spectrogram(vd, extra=True)
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
    #################get ground truth distribution ################################################################
    gt_skip_step = ECG_SAMPLE_RATE / FRAME_RATE
    vds = []
    gt_li = []
    for cond in [ 'lighting','movement']: 
        if cond == 'lighting':
            n = [0,1]
        else:
            n = [0,1,2]
        for i in n:
            _,vd = utils.create_file_paths(range(1,27), sensor_sgn=0,cond=cond, cond_typ=i)
            vds += vd
    for vd in vds:       
        gts = utils.cvt_sensorSgn(vd, gt_skip_step, extra=False)
        gt = np.mean(gts)
        gt_li.append(gt)
    gt_li = np.asarray(gt_li, dtype=np.float32)
    it1 = np.where(gt_li<50)[0]
    it2 = np.where(gt_li<60)[0]
    it3 = np.where(gt_li<70)[0]
    it4 = np.where(gt_li<80)[0]
    it5 = np.where(gt_li<90)[0]
    it6 = np.where(gt_li<100)[0]
    print(it1.shape[0])
    print((it2.shape[0]-it1.shape[0]))
    print((it3.shape[0]-it2.shape[0]))
    print((it4.shape[0]-it3.shape[0]))
    print((it5.shape[0]-it4.shape[0]))
    print((it6.shape[0]-it5.shape[0]))
            
            
            
            
            
            