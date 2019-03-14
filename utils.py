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

FRAME_RATE = 30.0
N_FRAME = 3600
N_CLIPS = 120
CLIP_SIZE = int(N_FRAME / N_CLIPS)
VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']
LABEL_PATHS = ['D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin']

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

def face_align(pre_frame,next_frame,name,idx, pre_nose, next_nose):
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


def cal_face_landmarks(video_paths):
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
            file_label.append(val)
        diff_mean = np.mean(file_label)
        diff_std = np.std(file_label)
        diff_sgn_li.append((diff_mean, diff_std))
    return cond, sgn_li, diff_sgn_li


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
    if val > 3:
        val = 3
    if val < -3:
        val = -3
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
    
#     if idx%1000 == 0:
#         if not os.path.exists('./n_processed_video/'+'diff/'):
#             os.makedirs('./n_processed_video/' + 'diff/')
#         mask = np.zeros_like(re,dtype=np.uint8)
#         mask[np.where(abs(re)>1)[0],np.where(abs(re)>1)[1],:]=255
#         cv2.imwrite( ('./n_processed_video/' +'diff/'+name+'-'+str(idx) + '.jpg'),mask)
        
    up_pos = np.where(re>3)
    low_pos = np.where(re<-3)
#     re[up_pos[0],up_pos[1],:] = 1
#     re[low_pos[0],low_pos[1],:] = -1
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

def get_startPoint(video_path):
    with open('StartingPoint.pickle', 'rb') as file:
        start_points = pickle.load(file)
    path = video_path.split('/')
    if len(path) > 6:
        cond = path[5].split('_')[0]
        u_id = int(path[4][-1])
        t_id = int(path[4][-2])
        prob_id = t_id * 10 + u_id - 1
    else:
        cond = '301'
        prob_id = int(path[4])
    ecg_idx, vd_idx = start_points[cond][prob_id]
    file.close()
    return ecg_idx, vd_idx

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
    b, a = cheby2(order, rs, [low, high], btype='band')
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
    

#if __name__ == '__main__':

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
#     for cond in ['lighting', 'movement']:
#         if cond == 'lighting':
#             n = 6
#         else:
#             n = 4
#         for i in range(n):
#             _, lb = create_file_paths(np.arange(1, 27), cond=cond, cond_typ=i)
#             _, gt = create_file_paths(np.arange(1, 27), cond=cond, cond_typ=i, sensor_sgn=0)
#             con, col, diff_col = cal_meanStd_label(lb,gt)
#             dt[con] = col
#             diff_dt[con] = diff_col
#     _, lb = create_extra_file_paths(range(1,18))
#     con, col, diff_col = cal_meanStd_label(lb, lb)
#     dt[con] = col
#     diff_dt[con] = diff_col
#     with open('LabelMeanStddev.pickle', 'wb') as f:
#         pickle.dump(dt, f)
#     f.close()
#     with open('DiffLabelMeanStddev.pickle', 'wb') as f:
#         pickle.dump(diff_dt, f)
#     f.close()
# #############################check generated labels###########################################
#     with open('LabelMeanStddev.pickle', 'rb') as file:
#         l_mean_std = pickle.load(file)
#     for key,li in l_mean_std.items():
#         print(key+' '+str(len(li)))
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
####################################################################################################################
   




