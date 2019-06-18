#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import urllib
import PIL
from PIL import ImageOps, Image
import numpy as np
import pickle
import cv2
import struct
import math
import scipy
from scipy import fftpack
from scipy.signal import butter, cheby2, lfilter
from scipy import interpolate
import scipy.signal as signal
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import dlib
import face_recognition
import face_alignment
from scipy.interpolate import CubicSpline
from jeanCV import skinDetector 
import tensorflow as tf
import re
from Deeplab import DeepLabModel

FRAME_RATE = 30.0
N_FRAME = 3600
N_CLIPS = 120
CLIP_SIZE = int(N_FRAME / N_CLIPS)
VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']
LABEL_PATHS = ['D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin']
MODEL_PATH = '/Vitalcam_Dataset/FaceRegionDetection/deeplab/datasets/pascal_voc_seg/crop_512/iter_5000/train_on_trainval_set/export/frozen_inference_graph.pb'

def download(down_link, file_path, expected_bytes):
    if os.path.exists(file_path):
        print("model exists. No need to dwonload.")
        return
    print("downloading......")
    model_file, _ = urllib.request.urlretrieve(down_link, file_path)
    assert os.stat(model_file).st_size == expected_bytes
    print("successfully downloaded.")


def create_file_paths(probs, cond='lighting', cond_typ=0, sensor_sgn=1):
    v_src_path = '/Vitalcam_Dataset/07_Datenbank_Smarthome/Testaufnahmen/Proband'
    l_src_path = '/Vitalcam_Dataset/07_Datenbank_Smarthome/newSync/Proband'
    conditions = {'lighting': ['/101_natural_lighting', '/102_artificial_lighting',
                               '/103_abrupt_changing_lighting', '/104_dim_lighting_auto_exposure',
                               '/106_green_lighting', '/107_infrared_lighting'],
                  'movement': ['/201_shouldercheck', '/202_scale_movement', '/203_translation_movement',
                               '/204_writing']}
    video_name = '/Logitech HD Pro Webcam C920.avi'
    label_name = '/synced2_Logitech HD Pro Webcam C920/'
    sgn_typ = ['6_Pulse.bin', '5_Pleth.bin', '1_EKG-AUX.bin']

    video_paths = []
    label_paths = []

    for i in probs:
        prob_id = str(i) if (i > 9) else ('0' + str(i))
        video_path = v_src_path + prob_id + conditions[cond][cond_typ] + video_name
        label_path = l_src_path + prob_id + conditions[cond][cond_typ] + label_name + sgn_typ[sensor_sgn]
        video_paths.append(video_path)
        label_paths.append(label_path)
    # print(video_paths)
    # print(label_paths)
    return video_paths, label_paths

def create_extra_file_paths(probs):
    video_paths = []
    label_paths = []
    for i in probs:
        video_name = [f for f in os.listdir('/Vitalcam_Dataset/10_Daten-Arne/Subjects/{}/'.format(i)) if f.endswith('.avi')][0]
        label_name = [f for f in os.listdir('/Vitalcam_Dataset/10_Daten-Arne/Subjects/{}/'.format(i)) if f.startswith('pulse')][0]
        video_path = '/Vitalcam_Dataset/10_Daten-Arne/Subjects/{}/'.format(i) + video_name
        label_path = '/Vitalcam_Dataset/10_Daten-Arne/Subjects/{}/'.format(i) + label_name
        video_paths.append(video_path)
        label_paths.append(label_path)
    #print(video_paths)
    #print(label_paths)
    return video_paths, label_paths


def detect_face(image):
    min_size = (30, 30)
    haar_scale = 1.2
    min_neighbors = 5
    haar_flags = 0
    faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.equalizeHist(temp, temp)

    faces = faceCascade.detectMultiScale(
        temp,
        haar_scale, min_neighbors, haar_flags, min_size
    )
    return faces

def face_align(pre_frame,next_frame,name,idx, pre_nose, next_nose):#no use
    t_x = pre_nose[0]-next_nose[0] if abs(pre_nose[0]-next_nose[0])>1 else 0
    t_y = pre_nose[1]-next_nose[1] if abs(pre_nose[1]-next_nose[1])>1 else 0    
    if abs(t_x)==0 and abs(t_y)==0:
        return pre_frame, next_frame 
    
    M = np.float32([[1,0,t_x],[0,1,t_y]])
    cols,rows = pre_frame.shape[:2]
    next_frame = cv2.warpAffine(next_frame,M,(rows,cols), flags=cv2.INTER_CUBIC) 
    pre_frame = pre_frame[max(0,t_y):min(cols,(cols+t_y)),max(0,t_x):min(rows,(rows+t_x)),:]
    next_frame = next_frame[max(0,t_y):min(cols,(cols+t_y)),max(0,t_x):min(rows,(rows+t_x)),:]  
    if not os.path.exists('./n_processed_video/'+'test/'):
        os.makedirs('./n_processed_video/' + 'test/')
    if idx%10 == 0:
        cv2.imwrite( ('./n_processed_video/' +'test/'+name+'-'+str(idx) + '.jpg'),pre_frame)
        cv2.imwrite( ('./n_processed_video/' +'test/'+name+'-'+ str(idx) + '-(1).jpg'),next_frame)
    return pre_frame, next_frame


def cal_face_landmarks(video_paths): #no use
    for v_path in video_paths:
        print(v_path)
        print(os.path.exists(v_path))
        path = v_path.split('/')
        prob_id = path[4]
        cond = path[5].split('_')[0]
        print(cond + '-' + prob_id)
        vd = {}
        for clip in range(1, int(N_CLIPS + 1)):
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
                frame = cv2.imread(pre_path)
                face_landmarks = face_recognition.face_landmarks(frame,model="small") 
                if len(face_landmarks)==0 or len(face_landmarks[0]["nose_tip"])==0:
                    nose = []
                    print('no landmarks!!! - ' + str(idx))
                else:
                    nose = face_landmarks[0]["nose_tip"][0]
                vd[str(idx)] = nose
        with open('./processed_video/' + cond + '/' + prob_id + '/'+'Landmarks.pickle', 'wb') as f:
            pickle.dump(vd, f)
        f.close() 
            
            
def get_face_landmarks(video_path):
    path = video_path.split('/')
    prob_id = path[4]
    cond = path[5].split('_')[0]
    scr_path = './processed_video/' + cond + '/' + prob_id + '/' 
    with open(scr_path+'Landmarks.pickle', 'rb') as file:
        landmarks = pickle.load(file)
    file.close()
    return landmarks


def skin_seg(video_path,width, height, resized=True):    
    mask_path = video_path[:20]+'mask/'+video_path[20:]
    seg_map = cv2.imread(mask_path,0)
    if resized :
        seg_map = cv2.resize(seg_map, (width, height))
    return seg_map.astype(np.uint8)

def fcn_nn(frame_batch):#no use
    m_frames = []
    seg_model = DeepLabModel(MODEL_PATH)
    for frame in frame_batch:
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        _, seg_map = seg_model.run(image)
        seg_map = seg_map.astype(np.uint8)
        frame = tf.bitwise_and(frame,seg_map)
        m_frames.append(frame)
    m_frames = np.asarray(m_frames, dtype=np.float32)
    return m_frames

def kl_divergence(p, p_es):
    return p * tf.log(p) - p * tf.log(p_es) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_es)

def compute_chrom(trace_signal): 
    b = trace_signal[:,0]
    g = trace_signal[:,1]
    r = trace_signal[:,2]
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b
    x_f = ppg_filt(x,40,200)
    y_f = ppg_filt(y,40,200)
    std_x = np.asarray(x_f, dtype=np.float32).std()
    std_y = np.asarray(y_f, dtype=np.float32).std()
    a = std_x / std_y
    s = x_f - a * y_f
    return s

def plot_signal(file_name):
    ref_s = []
    pred_s = []
    for line in open(file_name):
        if not re.match("\d|-", line):
            continue
        pred = re.split('( )+',line)[0]
        ref = re.split('( )+',line)[2]
        ref_s.append(ref)
        pred_s.append(pred)
    pred_s = butter_bandpass_filter(np.asarray(pred_s, dtype=np.float32), lowcut=1, highcut=2.5, fs=30, order=6).astype(np.float32)
    pred_s = (pred_s - np.mean(pred_s)) / np.std(pred_s)
    ref_s = butter_bandpass_filter(np.asarray(ref_s, dtype=np.float32), lowcut=0.75, highcut=4, fs=30, order=6).astype(np.float32)
    t = np.linspace(0., 5., 5*30 )
    s_pos = 30*60
    e_pos = 30*60 + 30*5 
    plt.xlabel('Seconds')
    plt.plot(t, ref_s[s_pos:e_pos], 'r-',label='referance')
    plt.plot(t, pred_s[s_pos:e_pos], 'k-',label='pred')
    plt.legend(loc='upper right')
    plt.savefig(file_name[30:43]+".png")
        
def plot_bss_signal(file_name):
    s1 = []
    s2 = []
    s3 = []
    for line in open(file_name):
        x1 = re.split('( )+',line)[0]
        x2 = re.split('( )+',line)[2]
        x3 = re.split('( )+',line)[4][:-2]
        s1.append(x1)
        s2.append(x2)
        s3.append(x3)

    s1 = butter_bandpass_filter(np.asarray(s1, dtype=np.float32), lowcut=0.75, highcut=4, fs=30, order=6).astype(np.float32)
    s1 = (s1 - np.mean(s1)) / np.std(s1)
    s2 = butter_bandpass_filter(np.asarray(s2, dtype=np.float32), lowcut=0.75, highcut=4, fs=30, order=6).astype(np.float32)
    s2 = (s2 - np.mean(s2)) / np.std(s2)
    s3 = butter_bandpass_filter(np.asarray(s3, dtype=np.float32), lowcut=0.75, highcut=4, fs=30, order=6).astype(np.float32)
    s3 = (s3 - np.mean(s3)) / np.std(s3)
    t = np.linspace(0., 5., 5*30 )
    s_pos = 30*60
    e_pos = 30*60 + 30*5 
    plt.xlabel('Seconds')
    plt.subplot(3,1,1)
    plt.plot(t, s1[s_pos:e_pos], 'r-',label='s1')
    plt.subplot(3,1,2)
    plt.plot(t, s2[s_pos:e_pos], 'k-',label='s2')
    plt.subplot(3,1,3)
    plt.plot(t, s3[s_pos:e_pos], 'c-',label='s3')
    plt.legend(loc='upper right')
    plt.savefig(file_name[30:43]+".png")
        

def cal_meanStd_video(video_paths, width=64, height=64):
    col = []
    for v_path in video_paths:
        frame_li = []
        mean = 0
        std = 1
        print(v_path)
        print(os.path.exists(v_path))
        path = v_path.split('/')
        if len(path) > 6:
            cond = path[5].split('_')[0]
            prob_id = path[4]
            n_clips = 120
        else:
            cond = '301'
            if int(path[4])>9:
                prob_id = 'Proband' + path[4]
            else:
                prob_id = 'Proband0' + path[4]
            n_clips = 600
        print(cond + '-' + prob_id)
        for clip in range(1, int(n_clips + 1)):
            scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'            
            start_pos = (clip - 1) * CLIP_SIZE
            end_pos = clip * CLIP_SIZE
            print(cond + '-' + prob_id + '-clip' + str(clip))
            for idx in range(start_pos, end_pos): 
                if idx % 100 == 0:
                    print("reading in frame " + str(idx) + "," + str(idx + 1))
                pre_path = scr_path + str(idx) + '.jpg'                
                if not os.path.exists(pre_path):
                    print('no such frame!')
                    continue
                frame = cv2.imread(pre_path).astype(np.float32) 
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                frame_li.append(frame)
        print('done adding!!!!!!!!!!!! '+str(len(frame_li)))
        frame_li = np.asarray(frame_li, dtype=np.float32)
        print(frame_li.shape)
        if len(frame_li) == 0:
            mean = np.array([0,0,0])
            std = np.array([1,1,1])
        else:
            mean = np.mean(frame_li,axis=(0,1,2))
            std = np.std(frame_li,axis=(0,1,2))
        print('done calculating!!!!!!!!!!!!')
        print(mean)
        print(std)
        col.append((mean, std))
    return cond, col


def cal_meanStd_vdiff(video_paths, width=64, height=64):
    col = []
    for video_path in video_paths:
        li = []
        print(video_path)
        print(os.path.exists(video_path))
        path = video_path.split('/')
        if len(path) > 6:
            cond = path[5].split('_')[0]
            prob_id = path[4]
            n_clips = 120
        else:
            cond = '301'
            if int(path[4])>9:
                prob_id = 'Proband' + path[4]
            else:
                prob_id = 'Proband0' + path[4]
            n_clips = 600
        for clip in range(1, int(n_clips + 1)):
            scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
            start_pos = (clip - 1) * CLIP_SIZE
            end_pos = clip * CLIP_SIZE
            print(cond + '-' + prob_id + '-clip' + str(clip))
            for idx in range(start_pos, end_pos):
                if idx % 100 == 0:
                    print("reading in frame " + str(idx) + "," + str(idx + 1))
                pre_path = scr_path + str(idx) + '.jpg'
                if idx == end_pos - 1 and clip != n_clips:
                    print('end of clip-' + str(clip))
                    print('reading in ' + str((clip) * CLIP_SIZE) + '.jpg')
                    scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip + 1) + '/'
                    next_path = scr_path + str(clip * CLIP_SIZE) + '.jpg'
                elif idx == end_pos - 1 and clip == n_clips:
                    print('done ' + cond + '-' + prob_id)
                    continue
                else:
                    next_path = scr_path + str(idx + 1) + '.jpg'
                if not ( os.path.exists(pre_path) and os.path.exists(next_path)):
                    print('no such frame!')
                    continue
                pre_frame = cv2.imread(pre_path).astype(np.float32)
                next_frame = cv2.imread(next_path).astype(np.float32)
                pre_frame = cv2.resize(pre_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                next_frame = cv2.resize(next_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                diff = np.subtract(next_frame, pre_frame)
                mean_fr = np.add(next_frame, pre_frame)
                re = np.true_divide(diff, mean_fr, dtype=np.float32)
                re[re == np.inf] = 0
                re = np.nan_to_num(re)
                li.append(re)
        if len(li) == 0:
            mean = np.array([0,0,0])
            std = np.array([1,1,1])
        else:
            mean = np.mean(li, axis=(0,1,2))
            std = np.std(li, axis=(0,1,2))
        print(mean)
        print(std)
        col.append((mean, std))
    return cond, col


def cal_meanStd_sgram(video_paths, width=64, height=64):#no use
    sgram_li = []
    for v_path in video_paths:
        frame_li = []
        print(v_path)
        print(os.path.exists(v_path))
        path = v_path.split('/')
        if len(path) > 6:
            cond = path[5].split('_')[0]
            prob_id = path[4]
            n_clips = 120
        else:
            cond = '301'
            if int(path[4])>9:
                prob_id = 'Proband' + path[4]
            else:
                prob_id = 'Proband0' + path[4]
            n_clips = 600
        print(cond + '-' + prob_id)
        for clip in range(1, int(n_clips + 1)):
            scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'            
            start_pos = (clip - 1) * CLIP_SIZE
            end_pos = clip * CLIP_SIZE
            print(cond + '-' + prob_id + '-clip' + str(clip))
            for idx in range(start_pos, end_pos): 
                if idx % 100 == 0:
                    print("reading in frame " + str(idx) + "," + str(idx + 1))
                pre_path = scr_path + str(idx) + '.jpg'                
                if not os.path.exists(pre_path):
                    print('no such frame!')
                    continue
                frame = cv2.imread(pre_path).astype(np.float32) 
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                mask = skin_seg(pre_path,width, height)
                frame = cv2.bitwise_and(frame,frame,mask = mask)       
                frame_li.append(frame)
        print('done adding!!!!!!!!!!!! '+str(len(frame_li)))
        window_size = 120
        for i in range(0,len(frame_li)- window_size + 1,15): 
            if not np.any(frame_li[i: i + window_size], axis=(1,2,3)).all():
                print('batch contains None Frame!')
                continue
            spec_gram = get_spectrogram(frame_li[i: i + window_size])
            sgram_li.append(spec_gram)
    
    print('done calculating!!!!!!!!!!!!')
    print( np.average(sgram_li,0))
    print(np.std(sgram_li,0))    




def cal_meanStd_label(label_paths,gt_paths):
    sgn_li = []
    diff_sgn_li = []
    skip_step = 256.0 / 30.0
    gt_skip_step = 16.0 / 30.0
    for label_path, gt_path in zip(label_paths, gt_paths):
        file_label = []        
        print(os.path.exists(label_path))
        path = label_path.split('/')
        if len(path) > 6:
            cond = path[5].split('_')[0]
            prob_id = path[4]
            labels = cvt_sensorSgn(label_path, skip_step)               
            gts = cvt_sensorSgn(gt_path, gt_skip_step)
        else:
            cond = '301'
            if int(path[4])>9:
                prob_id = 'Proband' + path[4]
            else:
                prob_id = 'Proband0' + path[4]
            labels = cvt_sensorSgn(label_path, skip_step, extra=True)               
            gts = cvt_sensorSgn(gt_path, gt_skip_step, extra=True)
        
        labels = ppg_filt(labels,min(gts),max(gts))
        mean = np.mean(labels)
        std = np.std(labels)
        sgn_li.append((mean, std))
        
        for idx in range(len(labels) - 1):
            val = float(labels[idx + 1] - labels[idx])
#         for idx in range(len(labels) - 1):
#             val = float(labels[idx ] - labels[idx + 1])
#             file_label.append(val)
        diff_mean = np.mean(file_label)
        diff_std = np.std(file_label)
        diff_sgn_li.append((diff_mean, diff_std))
    return cond, sgn_li, diff_sgn_li

def cal_meanStd_gt(label_paths,gt_paths):
    sgn_li = []
    diff_li = []
    gt_skip_step = 16.0 / 30.0
    for label_path, gt_path in zip(label_paths, gt_paths):
#         file_label = []        
        print(os.path.exists(label_path))
        path = label_path.split('/')
        if len(path) > 6:
            cond = path[5].split('_')[0]
            prob_id = path[4]             
            gts = cvt_sensorSgn(gt_path, gt_skip_step)
        else:
            cond = '301'
            if int(path[4])>9:
                prob_id = 'Proband' + path[4]
            else:
                prob_id = 'Proband0' + path[4]  
            gts = cvt_sensorSgn(gt_path, gt_skip_step, extra=True)
        print(cond+'-'+prob_id)
        sgn_li += list(gts)
    mean = np.mean(sgn_li)
    std = np.std(sgn_li)
#     sgn_li.append((mean, std))
    print(mean)
    print(std)

#     return cond, sgn_li, diff_li


def ppg_filt(labels,l_hr,h_hr):
    fs_Hz = 30  
    h_fund_Hz = h_hr/60*1.05
    l_fund_Hz = max(0.7,l_hr/60*0.95)
    labels = butter_bandpass_filter(labels, l_fund_Hz, h_fund_Hz, fs_Hz, 4)
    return labels


def rescale_label(val,label_path,mode='diff_label'):    
    mean, std = get_meanstd(label_path, mode=mode)
    val = val - mean
    val = val / std 
    if val > 2:
        val = 2
    if val < -2:
        val = -2
    return val


def norm_frame(img, mean=0, dev=1.0):   
    mean = mean.reshape((1, 1, 3))
    dev = dev.reshape((1, 1, 3))
    img = img - mean
    img = np.true_divide(img, dev)
    return img


def clip_dframe(re,name,idx, mean=0, dev=1.0):
    mean = mean.reshape((1, 1, 3))
    dev = dev.reshape((1, 1, 3))
    re = re - mean
    re = np.true_divide(re, dev)      
    up_pos = np.where(re>3)
    low_pos = np.where(re<-3)
    re[up_pos] = 3
    re[low_pos] = -3
    return re


def get_meanstd(video_path, mode='video'):
    if mode == 'video':
        with open('MeanStddev.pickle', 'rb') as file:
            mean_std = pickle.load(file)
    elif mode == 'diff':
        with open('DiffFrameMeanStddev.pickle', 'rb') as file:
            mean_std = pickle.load(file)
    elif mode == 'label':
        with open('LabelMeanStddev.pickle', 'rb') as file:
            mean_std = pickle.load(file)
    elif mode == 'diff_label':
        with open('DiffLabelMeanStddev.pickle', 'rb') as file:
            mean_std = pickle.load(file)
    elif mode == 'revdiff_label':
        with open('RevDiffLabelMeanStddev.pickle', 'rb') as file:
            mean_std = pickle.load(file)
    elif mode == 'HR':
        with open('HRMeanStddev.pickle', 'rb') as file:
            mean_std = pickle.load(file)
            
    path = video_path.split('/')
    if len(path) > 6:
        cond = path[5].split('_')[0]
        u_id = int(path[4][-1])
        t_id = int(path[4][-2])
        prob_id = t_id * 10 + u_id - 1
    else:
        cond = '301'
        prob_id = int(path[4])
    mean, dev = mean_std[cond][prob_id]
    file.close()
    return mean, dev

def get_delay(video_path):
    with open('DelayTime.pickle', 'rb') as file:
        delay_time = pickle.load(file)
    path = video_path.split('/')    
    if len(path) > 6:
        cond = path[5].split('_')[0]
        u_id = int(path[4][-1])
        t_id = int(path[4][-2])
        prob_id = t_id * 10 + u_id - 1
    else:
        cond = '301'
        prob_id = int(path[4])
    lag = delay_time[cond][prob_id]
    file.close()
    return lag

def cvt_sensorSgn(label_path, skip_step, data_len=8, extra=False):
    labels = []    
    if not extra:
        binFile = open(label_path, 'rb')
        try:
            while True:
                sgn = binFile.read(data_len)
                d_sgn = struct.unpack("d", sgn)[0]
                labels.append(d_sgn)
        except Exception:
            pass
        binFile.close()
        x = np.arange(0, len(labels))
        f = interpolate.interp1d(x, labels,fill_value="extrapolate")
        xnew = np.arange(0, len(labels), skip_step)    
        sampled_labels = f(xnew) 
    else:
        
        if skip_step > 1:
            idx = -1
        else:
            idx = -2
        with open(label_path, 'r') as f:
            labels = np.array([k.strip('\n').split(",")[idx] for k in f.readlines()], dtype=float)   
        sampled_labels = CubicSpline(np.arange(0, len(labels))/59.88, labels)(np.arange(0, int(len(labels)*30/59.88))/30)
    
    
#     plt.gcf().clear()
#     plt.plot(x, labels, 'o', xnew, sampled_labels, 'o')
#     plt.savefig("interplot.png")
#     plt.gcf().clear()

    return sampled_labels


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data)
    return y


def cheby2_bandpass_filter(data, rs, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby2(order, fs, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def text_save(ppgs, video_path ,mode='a'):
    path = video_path.split('/')
    prob_id = path[4]
    cond = path[5].split('_')[0]
    filename=cond+'-'+prob_id
    scr_path = './pred'
    file = open(scr_path+filename,mode)
    for i in range(len(ppgs)):
        file.write(str(ppgs[i])+'\n')
    file.close()
    

if __name__ == '__main__':
#     plot_signal('./[result]Auto_encoder/prob01/101-Proband01-ppg(33).txt')
    plot_bss_signal('./predict_results/101-Proband15-bss(1).txt')
    #######remote&whole#######mean&std video file####################################################
#     dt = {}
#     for cond in ['lighting','movement']:
#         if cond == 'lighting':
#             n = [0,1,3]
#         else:
#             n = [0,1,2]
#         for i in n:
#             vd, _ = create_file_paths(range(1, 27), cond=cond, cond_typ=i)
#             con, col = cal_meanStd_video(vd)
#             dt[con] = col
#     vd, _ = create_extra_file_paths(range(1,18))
#     con, col = cal_meanStd_video(vd)
#     dt[con] = col
#     for key,li in dt.items():
#         print(key+' '+str(len(li))) 
#     with open('MeanStddev.pickle', 'wb') as f:
#         pickle.dump(dt, f)
#     f.close()
                    
    #######remote&whole#######diff-frame mean&std file####################################################
#     dt = {}
#     for cond in ['lighting','movement']:
#         if cond == 'lighting':
#             n = [0,1,3]
#         else:
#             n = [0,1,2]
#         for i in n:
#             vd, _ = create_file_paths(range(1, 27), cond=cond, cond_typ=i)
#             con, col = cal_meanStd_vdiff(vd)
#             dt[con] = col
#     vd, _ = create_extra_file_paths(range(1,18))
#     con, col = cal_meanStd_vdiff(vd)
#     dt[con] = col
#     for key,li in dt.items():
#         print(key+' '+str(len(li))) 
#     with open('DiffFrameMeanStddev.pickle', 'wb') as f:
#         pickle.dump(dt, f)
#     f.close()

   #########################check whether can get mean&std##############################################
#     with open('MeanStddev.pickle', 'rb') as file:
#         v_mean_std = pickle.load(file)
#     with open('DiffFrameMeanStddev.pickle', 'rb') as f:
#         vd_mean_std = pickle.load(f)
#     for key,li in v_mean_std.items():
#         print(len(li))
#     for key,li in vd_mean_std.items():
#         print(len(li))
    ########remote mean&std labels#####################################################################
#     dt = {}
#     diff_dt = {}
#     lb = []
#     gt = []
#     for cond in ['lighting', 'movement']:
#         if cond == 'lighting':
#             n = [0,1,3]
#         else:
#             n = [0,1]
#         for i in n:
#             lb += create_file_paths(np.arange(1, 27), cond=cond, cond_typ=i)[-1]
#             gt += create_file_paths(np.arange(1, 27), cond=cond, cond_typ=i, sensor_sgn=0)[-1]
#     lb += create_extra_file_paths(range(1,18))[-1]
#     cal_meanStd_gt(lb,gt)
#     cal_meanStd_sgram(lb)
#             dt[con] = col
#             diff_dt[con] = diff_col
#     _, lb = create_extra_file_paths(range(1,18))
#     con, col, diff_col = cal_meanStd_gt(lb, lb)
#     dt[con] = col
#     diff_dt[con] = diff_col
#     with open('LabelMeanStddev.pickle', 'wb') as f:
#         pickle.dump(dt, f)
#     f.close()
#     with open('DiffLabelMeanStddev.pickle', 'wb') as f:
#         pickle.dump(diff_dt, f)
#     f.close()
#     with open('RevDiffLabelMeanStddev.pickle', 'wb') as f:
#         pickle.dump(diff_dt, f)
#     f.close()
#     with open('HRMeanStddev.pickle', 'wb') as f:
#         pickle.dump(dt, f)
#     f.close()
# #############################check generated labels###########################################
#     with open('LabelMeanStddev.pickle', 'rb') as file:
#         l_mean_std = pickle.load(file)
#     with open('HRMeanStddev.pickle', 'rb') as file:
#         l_mean_std = pickle.load(file)
#     s = 0
#     m = 0
#     l = 0
#     max = 1
#     min = 1000
#     x = []
#     for key,li in l_mean_std.items():
#         if key=='103' or key=='107' or key=='106' or key=='103':
#             continue
#         for i in range(len(li)):
#             if int(li[i][0])< 30:
#                 continue
#             x.append(int(li[i][0]))
#     print(str(s)+'-'+str(m)+'-'+str(l))
#     print(str(min)+'-'+str(max))
    
#     num_bins = 10
#     fig = plt.Figure()
#     n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
#     plt.savefig('histo.png')
#         for i in range(26):
#             print(key+'-'+str(i+1)+'-'+str(li[i][0]))
#     with open('DiffLabelMeanStddev.pickle', 'rb') as file:
#         l_mean_std = pickle.load(file)
#     for key,li in l_mean_std.items():
#         print(key+' '+str(len(li)))
#######remote&whole#######landmarks file####################################################
#     for cond in ['lighting','movement']:
#         if cond == 'lighting':
#             n = 6
#         else:
#             n = 4
#         for i in range(n):
#             vd, _ = create_file_paths(range(1, 27), cond=cond, cond_typ=i)
#             cal_face_landmarks(vd)
#####################test landmarks#############################################################
#     for cond in ['movement']:#,'movement']:
#         if cond == 'lighting':
#             n = [2]
#         else:
#             n = [0]
#         for i in n:
#             vds, _ = create_file_paths(range(1, 27), cond=cond, cond_typ=i)
#             for video_path in vds:
#                 path = video_path.split('/')
#                 prob_id = path[4]
#                 cond = path[5].split('_')[0]
#                 landmarks = get_face_landmarks(video_path)
#                 for clip in range(1, int(N_CLIPS + 1)):
#                     scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
#                     start_pos = (clip - 1) * CLIP_SIZE
#                     end_pos = clip * CLIP_SIZE
#                     print(cond + '-' + prob_id + '-clip' + str(clip))
#                     for idx in range(start_pos, end_pos):
#                         if not idx%300 == 0:
#                             continue
#                         pre_path = scr_path + str(idx) + '.jpg'
#                         if idx == end_pos - 1 and clip != N_CLIPS:
#                             print('end of clip-' + str(clip))
#                             #print('reading in ' + str((clip) * CLIP_SIZE) + '.jpg')
#                             scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip + 1) + '/'
#                             next_path = scr_path + str(clip * CLIP_SIZE) + '.jpg'
#                         elif idx == end_pos - 1 and clip == N_CLIPS:
#                             print('done ' + cond + '-' + prob_id)
#                             raise StopIteration
#                         else:
#                             next_path = scr_path + str(idx + 1) + '.jpg'
#                         if not (os.path.exists(next_path) and os.path.exists(pre_path)):
#                             continue            
#                         pre_frame = cv2.imread(pre_path)#.astype(np.float32)
#                         next_frame = cv2.imread(next_path)#.astype(np.float32)
#                         pre_nose = landmarks[str(idx)]
#                         next_nose = landmarks[str(idx+1)]
#                         if len(pre_nose)==0 or len(next_nose)==0:
#                             continue
#                         cv2.circle(pre_frame,pre_nose,1,(55,255,155),1)                         
#                         cv2.circle(next_frame,next_nose,1,(55,255,155),1) 
#                         if not os.path.exists('./n_processed_video/'+'landmarks/'):
#                             os.makedirs('./n_processed_video/' + 'landmarks/')
#                         cv2.imwrite( ('./n_processed_video/' +'landmarks/'+prob_id+'-'+str(idx) + '.jpg'),pre_frame)
#                         cv2.imwrite( ('./n_processed_video/' +'landmarks/'+prob_id+'-'+str(idx) + '.jpg'),next_frame)
                        
#############################adding the lags###########################################
#     dt = {}
#     dt['101']=[-10,21,17, -9 , -8 , -11 , -9 ,-10, -11,-12 ,-13,-12 ,24, -26,-11,-10, -10 ,-11,-10,-11,-9,-11,-13,-11,-10,-11]
#     dt['102']=[ -11,-14 , -11,-9,-10,-11,-11,-9,-12,-593,-13, -11 ,-14,12,-12,-10,-9,-11, -11,11,-1220, -10,1399 ,-12,-234,19]
#     dt['103']=[114,0,2,0,0,0, 0,0,0,0,0,0,0,0,11,0,1,13,0,0,0,564,0,0,0,0]
#     dt['104']=[ -11,19 ,-12,-11,-11 ,-11 ,-11, -12 ,-13,-12 ,-13,-10,-14 ,-10 ,13,-11,-11,-10,-11,-11 ,-11,-9,350 ,-11, 16, -12]
#     dt['201']=[-12,-14,-11,-1020,-1534,11,-40,-11,-12,14,17,-41,-16 ,-11,15, -9, -10,-11,12,12, -60 ,-11,-13,-10,-11,-11]
#     dt['202'] =[-10,-15,18,62,-9,-11,-13,-11,-10,-12,-11,-11,-13,-10,-12,-9,-11,-11,-11,-12,-10 ,-12,-937,-9,-11,-10]
#     dt['203']=[-11,-14,18,-1442,-10,-12,-13,-10,-12,-10,-11,14,22,-12,-11,-12,-11,-11,11,-11 ,-10,-10,44,-11,-12,-11]
#     dt['204']=[114,2,0,0,2,-2, 0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,566,4,587,0,0]
#     dt['301']=[-5,-9,-5,-7,2,-6,-7,-46,-6,-5,-5,-5,-5,-42,-4,-5,0]
#     for i in ['101','102','103','104','201','202','203','204','301']:
#         print(len(dt[i]))
#     with open('DelayTime.pickle', 'wb') as f:
#         pickle.dump(dt, f)
#     f.close()
# ##########################align the starting point#################################################################
#     dt = {}
#     dt['101']= [(0,0)] * 26
#     dt['102']=[(0,0)] * 26
#     dt['103']=[(0,0)] * 26
#     dt['104']=[(0,0)] * 26
#     dt['201']=[(0,0)] * 26
#     dt['202'] =[(0,0)] * 26
#     dt['203']=[(0,0)] * 26
#     dt['204']=[(0,0)] * 26
#     dt['301']=[(4631,370),(49409,18062),(0,0),(3112,129),(32971,578),(43146,18726),(2850,195),(0,0),(7275,166),(52540,18109),(6549,218),(3176,90),(48360,19372),(2792,148),(3797,88),(0,0),(0,0)]
#     for i in ['101','102','103','104','201','202','203','204','301']:
#         print(len(dt[i]))
#     with open('StartingPoint.pickle', 'wb') as f:
#         pickle.dump(dt, f)
#     f.close()
################testing skin segmentation###########################################################################
#     scr_path = './processed_video/101/Proband01/1/'            
#     pre_path = scr_path + '1.jpg' 
#     frame = cv2.imread(pre_path)
#     a = skin_seg(pre_path)
#     cv2.imwrite( ('./1.jpg'),cv2.bitwise_and(frame ,frame,mask = a))   




