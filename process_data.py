#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import tensorflow as tf
import cv2
import utils
import numpy as np
import math
import random
import struct
import scipy
from scipy import fftpack
import pickle
from scipy.signal import butter, cheby2, lfilter
from imblearn.over_sampling import RandomOverSampler, SMOTE

# import matplotlib.pyplot as plt

ECG_SAMPLE_RATE = 16.0
PLE_SAMPLE_RATE = 256.0
FRAME_RATE = 30.0
VIDEO_DUR = 120
N_FRAME = 3600
N_CLIPS = 120
CLIP_SIZE = int(N_FRAME / N_CLIPS)
VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']
LABEL_PATHS = ['D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin']
GT_PATHS = ['D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/6_Pulse.bin']


def cvt_hr(labels, duration, fs, lowcut, highcut, order):
    N = len(labels)
    t = np.linspace(0, duration, N)

    y = utils.butter_bandpass_filter(labels, lowcut, highcut, fs, order)
    # y = cheby2_bandpass_filter(labels, 20, lowcut, highcut, fs, order=4)

    # periodogram
    FFT2 = abs(scipy.fft(y, N))
    f2 = 20 * scipy.log10(FFT2)
    f2 = f2[range(int(N / 2))]  # remove mirrored part of FFT
    freqs2 = scipy.fftpack.fftfreq(N, t[1] - t[0])
    freqs2 = freqs2[range(int(N / 2))]  # remove mirrored part of FFT

    d = np.argmax(f2)

    # Plotting periodogram
    x1 = freqs2[d]
    y1 = max(f2)     

    HeartRate = freqs2[d] * 60
    return HeartRate


def test_hr(label_path, duration, fs, lowcut=0.7, highcut=2.5, order=6, data_len=8):
    binFile = open(label_path, 'rb')
    N = int(fs * duration)
    hr = []
    skip_step = PLE_SAMPLE_RATE / FRAME_RATE
    for idx in range(120 - duration):
        labels = []
        offset = fs * idx
        n = 0
        try:
            while n < N:
                pos = math.floor(offset + n * skip_step)
                binFile.seek(pos * data_len)
                sgn = binFile.read(data_len)
                d_sgn = struct.unpack("d", sgn)[0]  
                labels.append(d_sgn)
                n += 1
            re = cvt_hr(labels, duration, fs, lowcut, highcut, order)
            hr.append(re)
        except Exception:
            pass
    binFile.close()
    return hr


def get_hr(labels, batch_size, duration, fs, lowcut=0.75, highcut=4, order=6):
    N = int(fs * duration)
    hr = []
    #labels = utils.butter_bandpass_filter(labels, lowcut, highcut, fs, order)
    for idx in range((-batch_size), 0, 1):
        start_pos = idx - N
        end_pos = idx
        print('ppg fraction:' + str(start_pos) + '--' + str(end_pos))
        ppg = labels[start_pos:end_pos]
        hr.append(cvt_hr(ppg, duration, fs, lowcut, highcut, order))
    return hr


def nor_diff_face(video_path, width=112, height=112, extra=False):
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
    mean, dev = utils.get_meanstd(video_path)
    re_mean, re_dev = utils.get_meanstd(video_path, mode='diff')
    #landmarks = utils.get_face_landmarks(video_path)        
    for clip in range(1, int(n_clips + 1)):
        scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
        start_pos = (clip - 1) * CLIP_SIZE
        end_pos = clip * CLIP_SIZE       
        #print(cond + '-' + prob_id + '-clip' + str(clip))
        for idx in range(start_pos, end_pos):
            pre_path = scr_path + str(idx) + '.jpg'
            if idx == end_pos - 1 and clip != n_clips:
                scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip + 1) + '/'
                next_path = scr_path + str(clip * CLIP_SIZE) + '.jpg'
            elif idx == end_pos - 1 and clip == n_clips:
                raise StopIteration
            else:
                next_path = scr_path + str(idx + 1) + '.jpg'
            if not (os.path.exists(next_path) and os.path.exists(pre_path)):
                yield [], []
                continue
#             pre_nose = landmarks[str(idx)]
#             next_nose = landmarks[str(idx+1)]
#             if len(pre_nose) == 0 or len(next_nose)==0:
#                 yield [], []
#                 continue  
            pre_frame = cv2.imread(pre_path)#.astype(np.float32)
            next_frame = cv2.imread(next_path)#.astype(np.float32)
            #pr_pre_frame, pr_next_frame = utils.face_align(pre_frame, next_frame, cond+'-'+prob_id,idx, pre_nose, next_nose) 
            pr_pre_frame = cv2.resize(pre_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            pr_next_frame = cv2.resize(next_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            diff = np.subtract(pr_next_frame, pr_pre_frame)
            mean_fr = np.add(pr_next_frame , pr_pre_frame )
            re = np.true_divide(diff, mean_fr, dtype=np.float32)
            re[re == np.inf] = 0
            re = np.nan_to_num(re)
            re = utils.clip_dframe(re, cond+'-'+prob_id,idx,re_mean, re_dev)
            pr_pre_frame = utils.norm_frame(pr_pre_frame, mean, dev)         
            yield pr_pre_frame, re


def nor_diff_clip(video_path, clip=1, width=112, height=112,extra=False):
    ###########remote##########################################
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
    scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
    start_pos = (clip - 1) * CLIP_SIZE
    end_pos = clip * CLIP_SIZE 
    mean, dev = utils.get_meanstd(video_path, mode='video')
    re_mean, re_dev = utils.get_meanstd(video_path, mode='diff')
    #landmarks = utils.get_face_landmarks(video_path)
    #print(cond + '-' + prob_id + '-clip' + str(clip))
    for idx in range(start_pos, end_pos):        
        pre_path = scr_path + str(idx) + '.jpg'
        if idx == end_pos - 1 and clip != n_clips:
            scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip + 1) + '/'
            next_path = scr_path + str(clip * CLIP_SIZE) + '.jpg'
        elif idx == end_pos - 1 and clip == n_clips:
            raise StopIteration
        else:
            next_path = scr_path + str(idx + 1) + '.jpg'            
        if not (os.path.exists(next_path) and os.path.exists(pre_path)):
            yield [], []
            continue
#         pre_nose = landmarks[str(idx)]
#         next_nose = landmarks[str(idx+1)]
#         if len(pre_nose) == 0 or len(next_nose)==0:
#             yield [], []
#             continue   
        pre_frame = cv2.imread(pre_path)#.astype(np.float32)
        next_frame = cv2.imread(next_path)#.astype(np.float32)  
        #pr_pre_frame, pr_next_frame = utils.face_align(pre_frame,next_frame, cond+'-'+prob_id,idx, pre_nose, next_nose)  
        pr_pre_frame = cv2.resize(pre_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        pr_next_frame = cv2.resize(next_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        diff = np.subtract( pr_next_frame, pr_pre_frame )        
        mean_fr = np.add( pr_next_frame  , pr_pre_frame  )        
        re = np.true_divide(diff, mean_fr, dtype=np.float32)        
        re[re == np.inf] = 0
        re = np.nan_to_num(re)
        re = utils.clip_dframe(re, cond+'-'+prob_id,idx,re_mean, re_dev)        
        pr_pre_frame = utils.norm_frame(pr_pre_frame, mean, dev)
        #print('frame idx: '+str(idx))
        yield pr_pre_frame, re


######################################################################


def get_sample(video_path, label_path, gt_path, clip=1, width=112, height=112, mode='train',extra=False):
    if mode == 'train':
        diff_iterator = nor_diff_clip(video_path, clip=clip, width=width, height=height,extra=extra)
        gt_skip_step = ECG_SAMPLE_RATE / FRAME_RATE
        gts = utils.cvt_sensorSgn(gt_path, gt_skip_step, extra=extra)
        
        skip_step = PLE_SAMPLE_RATE / FRAME_RATE
        labels = utils.cvt_sensorSgn(label_path, skip_step, extra=extra)
        labels = utils.ppg_filt(labels,min(gts),max(gts))
        
        lag = -utils.get_delay(video_path)
        start_pos = (clip - 1) * CLIP_SIZE
        end_pos = clip * CLIP_SIZE         
        for idx in range(start_pos, end_pos):
            frame, diff = next(diff_iterator)
#             print('frame idx:'+str(lag + idx - vd_idx))
#             print('label idx:'+str(idx+ lag + 1 - ecg_idx))
            if len(frame) == 0 or len(diff) == 0 or (lag + idx) < 0 :
                continue
            if (lag + idx)>= len(labels)- 1:
                break                        
            gt = float(gts[idx ])
            label = float(labels[idx+ lag + 1] - labels[idx+lag ])           
            val = utils.rescale_label(label, label_path)
            #print('label idx: '+str(idx)+'+'+str(lag))
            #print(str(labels[idx+ lag + 1])+' - '+str(labels[idx+lag]))
            #print(str(val)+' - '+str(gt))      
            yield (frame, diff, val, gt)            
    else:
        diff_iterator = nor_diff_face(video_path, width=width, height=height,extra=extra)
        
        gt_skip_step = ECG_SAMPLE_RATE / FRAME_RATE
        gts = utils.cvt_sensorSgn(gt_path, gt_skip_step, extra=extra)
        
        skip_step = PLE_SAMPLE_RATE / FRAME_RATE
        labels = utils.cvt_sensorSgn(label_path, skip_step, extra=extra)
        labels = utils.ppg_filt(labels,min(gts),max(gts))
        
        lag = utils.get_delay(video_path)
        for idx in range(N_FRAME - 1):
            frame, diff = next(diff_iterator)
            if len(frame) == 0 or len(diff) == 0 or (lag + idx) < 0:
                continue
            if (lag + idx)>= len(labels) - 1:
                break
            gt = float(gts[idx])
            label = float(labels[idx+ lag + 1] - labels[idx + lag])
            val = utils.rescale_label(label,label_path)
            yield (frame, diff, val, gt)
    


def get_batch(video_paths, label_paths, gt_paths, clips, batch_size, width=112, height=112, mode='train'):
    frame_batch = []
    diff_batch = []
    label_batch = []
    gt_batch = []
    sample_li = []
    random.shuffle(clips)
    paths = list(zip(video_paths, label_paths, gt_paths))
    if mode == 'train':
        for clip in clips:
            random.shuffle(paths)
            sample_feat = []
            for (video_path, label_path, gt_path) in paths:
                path = video_path.split('/') 
                if len(path) > 6:
                    extra = False
                    if clip > 120:
                        continue
                else:
                    extra = True
                iterator = get_sample(video_path, label_path, gt_path, clip=clip, width=width, height=height, mode=mode,extra=extra)
                try:
                    while True:
                        (frame, diff, label, gt) = next(iterator)
                        sample_feat.append((frame, diff, label, gt))                       
                except StopIteration:
                    pass
            random.shuffle(sample_feat)
            print('sample len: '+str(len(sample_feat))) 
            for frame, diff, label, gt in sample_feat:                
                sample_li.append((frame, diff, label, gt))
                if len(sample_li) < batch_size :
                    continue
                #random.shuffle(sample_li)
                for (frame, diff, label, gt) in sample_li:
                    frame_batch.append(frame)
                    diff_batch.append(diff)
                    label_batch.append(label)
                    gt_batch.append(gt)
                yield frame_batch, diff_batch, label_batch, gt_batch
                frame_batch = []
                diff_batch = []
                label_batch = []
                gt_batch = []
                sample_li = []
    else:
        for (video_path, label_path, gt_path) in zip(video_paths, label_paths, gt_paths):
            path = video_path.split('/')    
            if len(path) > 6:
                extra = False
            else:
                extra = True
            iterator = get_sample(video_path, label_path, gt_path, width=width, height=height, mode=mode,extra=extra)
            try:
                while True:
                    while len(frame_batch) < batch_size:
                        frame, diff, label, gt = next(iterator)
                        frame_batch.append(frame)
                        diff_batch.append(diff)
                        label_batch.append(label)
                        gt_batch.append(gt)
                    yield frame_batch, diff_batch, label_batch, gt_batch
                    # print('done one batch.')
                    frame_batch = []
                    diff_batch = []
                    label_batch = []
                    gt_batch = []
            except StopIteration:
                continue
        

def get_sample_seq(video_path, label_path, gt_path,width=112, height=112,extra=False):
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
    mean, dev = utils.get_meanstd(video_path)
    
    gt_skip_step = ECG_SAMPLE_RATE / FRAME_RATE
    gts = utils.cvt_sensorSgn(gt_path, gt_skip_step, extra=extra)

    skip_step = PLE_SAMPLE_RATE / FRAME_RATE
    labels = utils.cvt_sensorSgn(label_path, skip_step, extra=extra)
    labels = utils.ppg_filt(labels,min(gts),max(gts))
    lag = utils.get_delay(video_path)
    
    frame_li = []
    label_li = []
    gt_li = []
    for clip in range(1, int(n_clips + 1)):
        scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
        start_pos = (clip - 1) * CLIP_SIZE
        end_pos = clip * CLIP_SIZE       
        #print(cond + '-' + prob_id + '-clip' + str(clip))
        for idx in range(start_pos, end_pos):
            pre_path = scr_path + str(idx) + '.jpg'            
            if (lag + idx) < 0:
                continue
            if (lag + idx)>= len(labels) - 1:
                break
            if not os.path.exists(pre_path):
                pr_frame = np.zeros((width, height,3))
            else:
                frame = cv2.imread(pre_path)#.astype(np.float32)            
                pr_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)            
                pr_frame = utils.norm_frame(pr_frame, mean, dev) 
            gt = float(gts[idx])
            label = float(labels[idx+ lag] )
            val = utils.rescale_label(label,label_path, mode='label')
            frame_li.append(pr_frame)
            label_li.append(val)
            gt_li.append(gt)
    return frame_li, label_li, gt_li

                
def get_seq_batch(video_paths, label_paths, gt_paths,batch_size, window_size, clips,mode,width=112, height=112):
    frame_batch = []
    label_batch = []
    gt_batch = []
    sample_batch = []
    random.shuffle(clips)
    paths = list(zip(video_paths, label_paths, gt_paths))
    for (video_path, label_path, gt_path) in paths:
        path = video_path.split('/')    
        if len(path) > 6:
            extra = False
        else:
            extra = True
        frame_li, label_li, gt_li= get_sample_seq(video_path, label_path, gt_path, width=width, height=height,extra=extra)
        for i in range(len(frame_li)- window_size): 
            if extra == False:
                sample_batch.append((frame_li[i: i + window_size],label_li[i: i + window_size],gt_li[i: i + window_size]))                    
            else:
                if not np.all(frame_li[i: i + window_size], axis=(1,2,3)).all():
                    print('batch contains None Frame!')
                    continue
                sample_batch.append((frame_li[i: i + window_size],label_li[i: i + window_size],gt_li[i: i + window_size]))
        if mode == 'train': 
            random.shuffle(sample_batch)
        for (frames,labels,gts) in sample_batch:
            frame_batch.append(frames)
            label_batch.append(labels)
            gt_batch.append(gts)
            if len(frame_batch) < batch_size:       
                continue
            yield frame_batch, label_batch, gt_batch
            frame_batch = []
            label_batch = []
            gt_batch = []
        sample_batch = []

                
                
                
def cvt_class(m):
    duration = 30
    all = []
    pos = 0
    neg = 0
    zero = 0
    for key, li in m.items():
        sgns = []
        rand = []
        hr_label = []
        print(key+':')
        for val, hr in li:
            hr_label.append(hr)
            if val > 1:
                val = 1
            if val < -1:
                val = -1
            all.append(val)
            sgns.append(val)
            if val > 0:
                pos += 1
            elif val < 0:
                neg += 1
            else:
                zero += 1
            seed = random.random()
            if seed > 0.85:
                if val != 0:
                    val = -val
                else:
                    val = seed
            rand.append(val)
        # plt.hist(sgns, bins=20)
        # plt.title("all label difference distribution")
        # plt.xlabel('value')
        # plt.ylabel('occurance')
        # plt.show()
        pred = []
        label = []
        gt = []
        for idx in range(120 - duration):
            hr = cvt_hr(sgns[(idx * 30):(idx * 30 + 30 * 30)], 30, 30, lowcut=0.7, highcut=2.5, order=6)
            pred_hr = cvt_hr(rand[(idx * 30):(idx * 30 + 30 * 30)], 30, 30, lowcut=0.7, highcut=2.5, order=6)
            pred.append(pred_hr)
            label.append(hr)
            gt.append(hr_label[(idx + 30) * 30])
        note = 0
        accur = 0
        for rate, g, ac in zip(gt, label, pred):
            if abs(rate - g) < 5:
                note += 1
            if abs(rate - ac) < 5:
                accur += 1
        print(str(note / len(pred)) + ' - ' + str(accur / len(pred)))
    mean = np.mean(all)
    std = np.std(all)
    print(mean)
    print(std)
    print(str(neg) + ' - ' + str(zero) + ' - ' + str(pos))
    ############local: check cvt hr & gts##########################################
    plt.hist(all, bins=20)
    plt.title("all label difference distribution")
    plt.xlabel('value')
    plt.ylabel('occurance')
    plt.show()

#if __name__ == '__main__':
#######################################################
#     tr_vd_path, tr_lb_path = utils.create_file_paths([2])
#     _, tr_gt_path = utils.create_file_paths([2], sensor_sgn=0)
# #     tr_vd_path, tr_lb_path = utils.create_extra_file_paths([1])
#     train_gen = get_batch(tr_vd_path, tr_lb_path,tr_gt_path, np.arange(1,5), 30, mode='train',extra=False)
#     idx = 0
#     print('<<<<<<<<<train gen>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#     try:
#         while True:
#             frames, diffs, labels, gts = next(train_gen)
#             print('batch-'+str(idx))
#             idx += 1
#             for (frame, diff, label, gt) in zip(frames, diffs, labels, gts):
#                 # cv2.imwrite(('frame'+ str(idx) + '.jpg'), frame)
#                 # cv2.imwrite(('diff'+ str(idx) + '.jpg'), diff)
#                 # cv2.imshow('face', frame)
#                 # cv2.imshow('diff', diff)
#                 print(str(label)+' - '+str(gt))
#                 pass
#     except StopIteration:
#         pass
# print('<<<<<<<<<test gen>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#test_gen = get_batch(VIDEO_PATHS, GT_PATHS, [1, 2], 500, mode='test')
# idx = 0
# try:
#     while True:
#         frames, diffs, labels = next(test_gen)
#         for (frame, diff, label) in zip(frames, diffs, labels):
#             print('batch-' + str(idx))
#             idx += 1
#     # cv2.imwrite(('frame'+ str(idx) + '.jpg'), frame)
#     # cv2.imwrite(('diff'+ str(idx) + '.jpg'), diff)
#     cv2.imshow('face', frame.astype(np.uint8))
#     # cv2.imshow('diff', diff)
# print(label)
#     cv2.waitKey(0)
#             break
# except StopIteration:
#     pass

############local:read in ppg##########################################
    # with open('Pleth.pickle', 'rb') as f:
    #     m = pickle.load(f)
    # f.close()
    # cvt_class(m)


