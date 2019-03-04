import os
import utils
import process_data
import numpy as np
import cv2
import pickle
import random
import scipy
from scipy import fftpack
import scipy.signal as signal
from scipy.signal import butter, cheby2, lfilter
import struct
import imblearn
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


LABEL_PATHS = 'D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin'
GT_PATHS = 'D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/6_Pulse.bin'

############check calculated mean&std##########################################
#with open('LabelMeanStddev.pickle', 'rb') as f:
# with open('LabelMeanStddev.pickle', 'rb') as f:
# with open('DiffFrameMeanStddev.pickle', 'rb') as f:
#     m = pickle.load(f)
#for li in m['101']:
#    print(li)
# print(m['101'][1])
# print(m['101'][12])
# print(m['101'][21])

# video_paths=['a','b','c','d']
# label_paths=['A','B','C','D']
# paths = list(zip(video_paths, label_paths))
############shuffle samples##########################################
# clips = [1,2,3,4,5]
# random.shuffle(clips)
# for clip in clips:
#     random.shuffle(paths)
#     for vp, lp in paths:
#         print(vp + ' ' + lp + ' ' + str(clip))
#random.shuffle(paths)
# print(len(paths))
# for (v, l, c) in paths:
#    print(v+' '+l+' '+str(c))
#############align the sugnal#################################################################
def test():
    tr_vd_paths = []
    tr_lb_paths = []
    tr_gt_paths = []
    for cond in ['lighting']:#, 'movement']:
        if cond == 'lighting':
            n = [0,1,3]#,0,3]
        else:
            n =[0, 1, 2]# 
        for i in n:
            tr_vd_path, tr_lb_path = utils.create_file_paths(np.arange(1,27), cond=cond, cond_typ=i)
            _, tr_gt_path = utils.create_file_paths(np.arange(1, 27), cond=cond, cond_typ=i, sensor_sgn=0)
            tr_vd_paths += tr_vd_path
            tr_lb_paths += tr_lb_path
            tr_gt_paths += tr_gt_path
            
    for video_path, label_path, gt_path in zip(tr_vd_paths, tr_lb_paths,tr_gt_paths):
        path = video_path.split('/')
        prob_id = path[4]
        cond = path[5].split('_')[0]
        diff_iterator = process_data.nor_diff_face(video_path, width=128, height=256)
        gt_skip_step = 16.0 / 30.0
        gts = utils.cvt_sensorSgn(gt_path, gt_skip_step)       
        skip_step = 256.0 / 30.0
        labels = utils.cvt_sensorSgn(label_path, skip_step)
        labels = labels - np.mean(labels)
        f_labels = utils.ppg_filt(labels,min(gts),max(gts))

############################################################################################
        plt.plot(np.arange(600),np.asarray(labels[:600]), '-')
        plt.plot(np.arange(600),np.asarray(f_labels[:600]), '-')
 #       min_pos = signal.argrelextrema(labels, np.less, order=1)
#         _, properties = signal.find_peaks(labels,prominence=[0.0001,None],width=[1,None])
#         thr = properties["prominences"].mean()
#         print(prob_id+' '+str(thr))
#         peaks, properties = signal.find_peaks(labels,prominence=[0.7*thr,None])
#         x =[]
#         y = []
#         for p in min_pos[0]:
#             if p>300 :
#                 break
#             x.append(p)
#             y.append(labels[p])
#         for p in peaks:
#             if p>300 :
#                 break
#             x.append(p)
#             y.append(labels[p])
#        plt.plot(x,y, 'o')
###############################################################################################
#         lag = utils.get_delay(video_path)
#         frame_li = []
#         signal_li = []
#         data = []
#         for idx in range(30*20):
#             frame, diff = next(diff_iterator)
#             if len(frame) == 0 or len(diff) == 0 or (lag + idx) < 0:
#                 continue
#             if (lag + idx) < 0:
#                 continue
#             if (lag + idx)>= len(labels) - 1:
#                 break
#             data.append((frame, diff))
#             label = float(labels[idx+ lag + 1] - labels[idx + lag])
#             val = utils.rescale_label(label,label_path)
#             frame_li.append(np.mean(diff))
#             signal_li.append(val*100) 
#             print(idx)
#             print(str(np.mean(diff))+' - '+str(val))
#         #frame_li = utils.butter_bandpass_filter(frame_li, 0.7, 2.5, 30, 6)
#         #print('###########mean:'+str(np.mean(frame_li)))
#         x = np.arange( min(90,len(frame_li)) )
#         #plt.plot(x,signal_li[:300], '-')
#         plt.plot(x, frame_li[:min(90,len(frame_li))], '-')
#         #plt.plot( x,signal_li[:200], '-', x,labels[lag:lag+200], '-')#x, frame_li[100:300],'-',
        if not os.path.exists('./n_processed_video/labels/' + cond + '/' ):
            os.makedirs('./n_processed_video/labels/' + cond + '/' )
        plt.savefig('./n_processed_video/labels/' + cond + '/' + prob_id + '-'+'labels.png')
        plt.gcf().clear()

        
def read_ppg(i,skip_step):
    b=np.array([5.120554179098604e-04, 0, -0.003072332507459, 0, 0.007680831268648,0,-0.010241108358197,0,0.007680831268648,0,-0.003072332507459,0,5.120554179098604e-04])
    a=np.array([1,-8.743872223710831,35.548420971063010,-88.976312277860913,1.528571802991954e+02,-1.900030139756585e+02,1.752782049794515e+02,-1.209264482427993e+02,61.926089843339625,-22.956237837558941,5.847697750958567,-0.919137374949478,0.067429689727660])
    filename = [f for f in os.listdir('/Vitalcam_Dataset/10_Daten-Arne/Subjects/{}/'.format(i)) if f.startswith('pulse')][0]
    if skip_step > 1:
        idx = -1
    else:
        idx = -2
    with open('/Vitalcam_Dataset/10_Daten-Arne/Subjects/{}/{}'.format(i, filename), 'r') as f:
        labels = np.array([k.strip('\n').split(",")[idx] for k in f.readlines()], dtype=float)
        print(labels[:10])    
    #labels = scipy.signal.filtfilt(b,a,lines)
    sampled_labels = CubicSpline(np.arange(0, len(labels))/59.88, labels)(np.arange(0, int(len(labels)*30/59.88))/30)
    print(sampled_labels[:20])  
        
    
if __name__ == '__main__':
    #test()
    read_ppg(1,2)

   



