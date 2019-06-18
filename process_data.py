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
import matplotlib.pyplot as plt

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
#         print('ppg fraction:' + str(start_pos) + '--' + str(end_pos))
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
    for clip in range(1, int(n_clips + 1)):
        scr_path = './n_processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
        start_pos = (clip - 1) * CLIP_SIZE
        end_pos = clip * CLIP_SIZE       
#         print(cond + '-' + prob_id + '-clip' + str(clip))
        for idx in range(start_pos, end_pos):
            pre_path = scr_path + str(idx) + '.jpg'
            if idx == end_pos - 1 and clip != n_clips:
                scr_path = './n_processed_video/' + cond + '/' + prob_id + '/' + str(clip + 1) + '/'
                next_path = scr_path + str(clip * CLIP_SIZE) + '.jpg'
            elif idx == end_pos - 1 and clip == n_clips:
                raise StopIteration
            else:
                next_path = scr_path + str(idx + 1) + '.jpg'
            if not (os.path.exists(next_path) and os.path.exists(pre_path)):
                yield [], []
                continue 
            pre_frame = cv2.imread(pre_path)#.astype(np.float32)
            next_frame = cv2.imread(next_path)#.astype(np.float32) 
            pr_pre_frame = cv2.resize(pre_frame, (width, height), interpolation=cv2.INTER_CUBIC)#.astype(np.float32)
            pr_next_frame = cv2.resize(next_frame, (width, height), interpolation=cv2.INTER_CUBIC)#.astype(np.float32)
            
            pre_mask = utils.skin_seg(pre_path,width, height)
            next_mask = utils.skin_seg(next_path,width, height)
            
#             diff = np.subtract(pr_next_frame.astype(np.float32), pr_pre_frame.astype(np.float32))
#             mean_fr = np.add(pr_next_frame.astype(np.float32) , pr_pre_frame.astype(np.float32) )
#             re = np.true_divide(diff, mean_fr, dtype=np.float32)
#             re[re == np.inf] = 0
#             re = np.nan_to_num(re)
#             re = utils.clip_dframe(re, cond+'-'+prob_id,idx,re_mean, re_dev)
#             pr_pre_frame = utils.norm_frame(pr_pre_frame, mean, dev)  
#             pr_next_frame = utils.norm_frame(pr_next_frame, mean, dev)

#             pr_pre_frame = cv2.bitwise_and(pr_pre_frame,pr_pre_frame,mask = pre_mask)
            m_next_frame = cv2.bitwise_and(pr_next_frame,pr_next_frame,mask = next_mask)
            
            yield pr_next_frame, m_next_frame# , re


def nor_diff_clip(video_path, clip=1, width=112, height=112,extra=False):
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
    scr_path = './n_processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
    start_pos = (clip - 1) * CLIP_SIZE
    end_pos = clip * CLIP_SIZE 
    mean, dev = utils.get_meanstd(video_path, mode='video')
    re_mean, re_dev = utils.get_meanstd(video_path, mode='diff')
#     print(cond + '-' + prob_id + '-clip' + str(clip))
    for idx in range(start_pos, end_pos):
#         print(cond + '-' + prob_id + '-clip' + str(clip)+'-'+str(idx))
        pre_path = scr_path + str(idx) + '.jpg'
        if idx == end_pos - 1 and clip != n_clips:
            scr_path = './n_processed_video/' + cond + '/' + prob_id + '/' + str(clip + 1) + '/'
            next_path = scr_path + str(clip * CLIP_SIZE) + '.jpg'
        elif idx == end_pos - 1 and clip == n_clips:
            raise StopIteration
        else:
            next_path = scr_path + str(idx + 1) + '.jpg'            
        if not (os.path.exists(next_path) and os.path.exists(pre_path)):
            yield [], [],[]
            continue 
        pre_frame = cv2.imread(pre_path)#.astype(np.float32)
        next_frame = cv2.imread(next_path)#.astype(np.float32)  
         
        pr_pre_frame = cv2.resize(pre_frame, (width, height), interpolation=cv2.INTER_CUBIC)#.astype(np.float32)
        pr_next_frame = cv2.resize(next_frame, (width, height), interpolation=cv2.INTER_CUBIC)#.astype(np.float32)
        
        pre_mask = utils.skin_seg(pre_path,width, height)
        next_mask = utils.skin_seg(next_path, width, height) 
        
#         diff = np.subtract( pr_next_frame.astype(np.float32), pr_pre_frame.astype(np.float32) )        
#         mean_fr = np.add( pr_next_frame.astype(np.float32)  , pr_pre_frame.astype(np.float32)  )        
#         re = np.true_divide(diff, mean_fr, dtype=np.float32)        
#         re[re == np.inf] = 0
#         re = np.nan_to_num(re)
#         re = utils.clip_dframe(re, cond+'-'+prob_id,idx,re_mean, re_dev)        
#         pr_pre_frame = utils.norm_frame(pr_pre_frame, mean, dev)  
#         pr_next_frame = utils.norm_frame(pr_next_frame, mean, dev)  

#         pr_pre_frame = cv2.bitwise_and(pr_pre_frame,pr_pre_frame,mask = pre_mask)
        m_next_frame = cv2.bitwise_and(pr_next_frame,pr_next_frame,mask = next_mask)
#         if not os.path.exists('./test/'+ '/' + prob_id + '/'):
#             os.makedirs('./test/'+ '/' + prob_id + '/')
#         cv2.imwrite( ('./test/' + '/' + prob_id + '/'+ str(idx) + '.jpg'),pr_pre_frame) 
        yield pr_next_frame, m_next_frame,next_mask#, re
        


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
            pre_frame, next_frame,mask = next(diff_iterator)
            if len(pre_frame) == 0 or len(next_frame) == 0  or (lag + idx) < 0 :#or len(diff) == 0
                continue
            if (lag + idx)>= len(labels)- 1:
                break                        
            gt = float(gts[idx ])
            label = float(labels[idx+ lag + 1] - labels[idx+lag ]) 
            val = utils.rescale_label(label, label_path)
            pre_label = float( labels[idx+lag ] )        
            pre_val = utils.rescale_label(pre_label, label_path,mode = 'label')
            next_label = float( labels[idx+ lag + 1] )        
            next_val = utils.rescale_label(next_label, label_path,mode = 'label')
            yield (pre_frame, next_frame, mask, next_val, gt)

    else:
        diff_iterator = nor_diff_face(video_path, width=width, height=height,extra=extra)
        
        gt_skip_step = ECG_SAMPLE_RATE / FRAME_RATE
        gts = utils.cvt_sensorSgn(gt_path, gt_skip_step, extra=extra)
        
        skip_step = PLE_SAMPLE_RATE / FRAME_RATE
        labels = utils.cvt_sensorSgn(label_path, skip_step, extra=extra)
        labels = utils.ppg_filt(labels,min(gts),max(gts))
        
        lag = utils.get_delay(video_path)
        for idx in range(N_FRAME - 1):
            pre_frame, next_frame = next(diff_iterator)
#             if len(pre_frame) == 0 or len(next_frame) == 0 or (lag + idx) < 0: # or len(diff) == 0
#                 continue
            if (lag + idx)>= len(labels) - 1:
                break
            gt = float(gts[idx])
            label = float(labels[idx+ lag + 1] - labels[idx+lag ]) 
            val = utils.rescale_label(label, label_path)
            pre_label = float( labels[idx+lag ] )        
            pre_val = utils.rescale_label(pre_label, label_path,mode = 'label')
            next_label = float( labels[idx+ lag + 1] )        
            next_val = utils.rescale_label(next_label, label_path,mode = 'label')
            yield (pre_frame, next_frame, pre_val, next_val, gt)
            
    


def get_batch(video_paths, label_paths, gt_paths, clips, batch_size, width=112, height=112, mode='train'):    
    pre_frame_batch = []
    next_frame_batch = []
    diff_batch = []
    label_batch = []
    gt_batch = []
    random.shuffle(clips)
    paths = list(zip(video_paths, label_paths, gt_paths))
    if mode == 'train':
        for clip in clips:
            random.shuffle(paths)
            sample_feat = []
            print('clip:'+str(clip))
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
                        (pre_frame, next_frame,_, diff, label, gt) = next(iterator)
                        sample_feat.append((pre_frame, next_frame, diff, label, gt))                       
                except StopIteration:
                    pass
            random.shuffle(sample_feat)
            print('sample len: '+str(len(sample_feat))) 
            for pre_frame, next_frame, diff, label, gt in sample_feat:                                
                pre_frame_batch.append(pre_frame)
                next_frame_batch.append(next_frame)
                diff_batch.append(diff)
                label_batch.append(label)
                gt_batch.append(gt)
                if len(pre_frame_batch) < batch_size :
                    continue
                yield pre_frame_batch, next_frame_batch, diff_batch, label_batch, gt_batch
                pre_frame_batch = []
                next_frame_batch = []
                diff_batch = []
                label_batch = []
                gt_batch = []
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
                    pre_frame, next_frame, diff, label, gt = next(iterator)
                    pre_frame_batch.append(pre_frame)
                    next_frame_batch.append(next_frame)
                    diff_batch.append(diff)
                    label_batch.append(label)
                    gt_batch.append(gt)
                    if len(pre_frame_batch) < batch_size :
                        continue
                    yield pre_frame_batch, next_frame_batch, diff_batch, label_batch, gt_batch
#                     print('done one batch.')
                    pre_frame_batch = []
                    next_frame_batch = []
                    diff_batch = []
                    label_batch = []
                    gt_batch = []
            except StopIteration:
                pass
            
                
def get_seq_batch(video_paths, label_paths, gt_paths, clips,batch_size, window_size,width=112, height=112, mode='train'):
    if mode == 'train':
        step = window_size
    elif mode == 'test':
        step = window_size
    norm_frame_batch = []
    frame_batch = []
    mask_batch = []
    label_batch = []
    gt_batch = []
    sample_batch = []
    paths = list(zip(video_paths, label_paths, gt_paths))  
    random.shuffle(paths)
    for (video_path, label_path, gt_path) in paths:        
        path = video_path.split('/')    
        if len(path) > 6:
            extra = False            
        else:
            extra = True 
        norm_frame_li = []
        frame_li = []
        mask_li = []
        label_li = []
        gt_li = []
        for clip in clips:
            vd_li = []
            iterator = get_sample(video_path, label_path, gt_path,
                          clip=clip, width=width, height=height, mode=mode,extra=extra)
            try:
                while True:
                    pre_frame, next_frame, mask,label, gt = next(iterator)
                    if len(next_frame) == 0:
#                         norm_frame_li.append(np.zeros((1,3)))
#                         vd_li.append( np.zeros((28,28,3)) )
                        print('none frame!!')
                    else:
                        norm_frame_li.append(next_frame.mean(axis=(0,1)))
                        vd_li.append(pre_frame)
                        mask_li.append(mask)
                        label_li.append(label)
                        gt_li.append(gt)
            except StopIteration:
                pass
            if len(vd_li) == 0:
                continue
            for fr in vd_li:
                fr[fr == np.inf] = 0
                fr = np.nan_to_num(fr)
                frame_li.append(fr)
        if clip%12 == 0:
            print('clip len: '+str(len(frame_li)))            
        for i in range(0,len(frame_li),step):
            if not np.any(frame_li[i: i + window_size], axis=(1,2,3)).all():
                print('batch contains None Frame!')
                continue   
            if len(frame_li[i: i + window_size]) < window_size:
                break
            mean = np.asarray(norm_frame_li[i: i + window_size]).mean(axis=0) 
            norm_signal = (np.asarray(norm_frame_li[i: i + window_size])) / mean
            mean = np.asarray(norm_signal).mean(axis=0)
            std = np.asarray(norm_signal).std(axis=0)
            norm_signal = (norm_signal - mean ) / std
            norm_signal[norm_signal == np.inf] = 0
            norm_signal = np.nan_to_num(norm_signal)
#             for i ,j in zip(norm_signal,frame_li[i: i + window_size]):
#                 print(i)
#                 print(j.mean(axis=(0,1)))
#                 print('###############')
            sample_batch.append(( 
                         norm_signal,
                         frame_li[i: i + window_size],
                         mask_li[i: i + window_size],
                         label_li[i: i + window_size],
                         gt_li[i: i + window_size]))
            
    if mode == 'train':
        random.shuffle(sample_batch)
        
    for (norm_frames,frames,mask,labels,gts) in sample_batch:
        norm_frame_batch.append(norm_frames)
        frame_batch.append(frames)
        mask_batch.append(mask)
        label_batch.append(labels)
        gt_batch.append(gts)
        if len(frame_batch) < batch_size:       
            continue
        yield norm_frame_batch,frame_batch, mask_batch,label_batch, gt_batch
        norm_frame_batch = []
        frame_batch = []
        mask_batch = []
        label_batch = []
        gt_batch = []
    return

 

def get_mask_frame(video_path, clip=1, width=112, height=112,extra=False):
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
    scr_path = './n_processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
    start_pos = (clip - 1) * CLIP_SIZE
    end_pos = clip * CLIP_SIZE 
    mean, dev = utils.get_meanstd(video_path, mode='video')
    re_mean, re_dev = utils.get_meanstd(video_path, mode='diff')
#     print(cond + '-' + prob_id + '-clip' + str(clip))
    for idx in range(start_pos, end_pos):
        pre_path = scr_path + str(idx) + '.jpg'
        if idx == end_pos - 1 and clip == n_clips:
            raise StopIteration            
        if not os.path.exists(pre_path):
            continue 
        pre_frame = cv2.imread(pre_path)#.astype(np.float32)        
        pr_pre_frame = cv2.resize(pre_frame, (width, height), interpolation=cv2.INTER_CUBIC)#.astype(np.float32)
        pre_mask = utils.skin_seg(pre_path,width, height)
        yield pr_pre_frame, pre_mask

def get_mask_batch(video_paths, label_paths, gt_paths, clips, batch_size, width=112, height=112, mode='train'):    
    frame_batch = []
    mask_batch = []
    random.shuffle(clips)
    paths = list(zip(video_paths, label_paths, gt_paths))
    for clip in clips:
        random.shuffle(paths)
        sample_feat = []
#         print('clip:'+str(clip))
        for (video_path, label_path, gt_path) in paths:
            vd_fr_li = []
            vd_m_li = []
            path = video_path.split('/') 
            if len(path) > 6:
                extra = False
                if clip > 120:
                    continue
            else:
                extra = True
            iterator = get_mask_frame(video_path, clip=clip, width=width, height=height,extra=extra)
            try:
                while True:
                    (pre_frame, mask) = next(iterator)
                    vd_fr_li.append(pre_frame)
                    vd_m_li.append(mask)
            except StopIteration:
                pass
            vd_mean = np.mean(np.asarray(vd_fr_li, dtype=np.float32),axis=(0))
            vd_fr_li = np.asarray(vd_fr_li, dtype=np.float32) / vd_mean
            for fr,mask in zip(vd_fr_li,vd_m_li): 
                sample_feat.append((fr,mask))
        random.shuffle(sample_feat)
#         print('sample len: '+str(len(sample_feat))) 
        for pre_frame, mask in sample_feat:                                
            frame_batch.append(pre_frame)
            mask_batch.append(mask)
            if len(frame_batch) < batch_size :
                continue
            yield frame_batch, mask_batch, np.zeros((len(frame_batch),1)),np.zeros((len(frame_batch),1))
            frame_batch = []
            mask_batch = []
            
if __name__ == '__main__':
#######################################################
    tr_vd_path, tr_lb_path = utils.create_file_paths([6],cond='movement', cond_typ=0)
    _, tr_gt_path = utils.create_file_paths([6], sensor_sgn=0,cond='movement', cond_typ=0)
#     tr_vd_path, tr_lb_path = utils.create_extra_file_paths([1])
#     train_gen = get_batch(tr_vd_path, tr_lb_path,tr_gt_path, np.arange(1,2), 2, mode='train')
    train_gen = get_mask_batch(tr_vd_path, tr_lb_path,tr_gt_path, np.arange(41,43), batch_size = 10,mode='train')
    idx = 0
    print('<<<<<<<<<train gen>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    try:
        while True:
            frames,masks = next(train_gen)
            print('batch-'+str(idx))
            idx += 1 
            frames = np.asarray(frames)
               
#             for (frame, diff, label, gt) in zip(frames, diffs, labels, gts):
#                 pass
    except StopIteration:
        pass
#     print('<<<<<<<<<test gen>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#     tr_vd_path, tr_lb_path = utils.create_file_paths([1])
#     _, tr_gt_path = utils.create_file_paths([1], sensor_sgn=0)
# #     test_gen = get_batch(tr_vd_path, tr_lb_path, tr_gt_path ,np.arange(2, 601),32,mode='test')
#     test_seq_gen = get_seq_batch(tr_vd_path, tr_lb_path, tr_gt_path,[1], batch_size = 2,window_size=30, mode='test')
#     idx = 0
#     try:
#         while True:
#             print('done')
#             _, b_labels, b_gts = next(test_seq_gen)
#             print('batch-'+str(idx))
#             print(np.asarray(b_labels).shape)
#             idx += 1
# #             print(np.asarray(b_labels)[:,0].shape)
# #             b_labels = np.reshape(b_labels[:,0],(32,))
# #             b_gts = np.reshape(b_gts[:,0],(32,))
# #             for (l,bl,gt,bgt) in zip(labels,b_labels,gts,b_gts):
# #                 #print(str(l) +' '+ str(bl))
# #                 print(str(gt) +' '+str(bgt))
#     except StopIteration:
#         pass

############local:read in ppg##########################################
    # with open('Pleth.pickle', 'rb') as f:
    #     m = pickle.load(f)
    # f.close()
    # cvt_class(m)


