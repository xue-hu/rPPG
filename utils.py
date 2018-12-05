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


def detect_face(image):
    min_size = (20, 20)
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0
    faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.equalizeHist(temp, temp)

    faces = faceCascade.detectMultiScale(
        temp,
        haar_scale, min_neighbors, haar_flags, min_size
    )
    # if faces.size != 0 :
    #     for (x, y, w, h) in faces:
    #         # Convert bounding box to two CvPoints
    #         pt1 = (int(x), int(0.9*y))
    #         pt2 = (int(x + w), int(y + 1.8*h))
    #         cv2.rectangle(image, pt1, pt2, (255, 0, 0), 5, 8, 0)
    #         cv2.namedWindow('face',0)
    #         cv2.waitKey(0)
    #         cv2.imshow('face',image)
    return faces


def cal_meanStd_video(video_paths, width=256, height=256):
    col = []
    for v_path in video_paths:
        frame_li = []
        print(v_path)
        print(os.path.exists(v_path))
        path = v_path.split('/')
        prob_id = path[4]
        cond = path[5].split('_')[0]
        print(cond + '-' + prob_id)
        ###########local###########################################
        # prob_id = 'Proband02'
        # cond = '101'
        ###########################################################
        for clip in range(1, int(N_CLIPS + 1)):
            scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
            start_pos = (clip - 1) * CLIP_SIZE
            end_pos = clip * CLIP_SIZE
            print(cond + '-' + prob_id + '-clip' + str(clip))
            for idx in range(start_pos, end_pos):
                if idx % 100 == 0:
                    print("reading in frame " + str(idx) + "," + str(idx + 1))
                pre_path = scr_path + str(idx) + '.jpg'
                if not os.path.exists(pre_path):
                    continue
            #    print(os.path.exists(pre_path))
             #   print(os.path.exists(next_path))
                frame = cv2.imread(pre_path).astype(np.float32)
                frame_li.append(frame)
        mean = np.mean(frame_li, axis=(0,1,2))
        std = np.std(frame_li, axis=(0,1,2))
        print(mean.shape)
        print(std)
        col.append((mean, std))
    return cond, col


def cal_meanStd_vdiff(video_paths, width=256, height=256):
    col = []
    for video_path in video_paths:
        li = []
        print(video_path)
        print(os.path.exists(video_path))
        path = video_path.split('/')
        prob_id = path[4]
        cond = path[5].split('_')[0]
        ###########local###########################################
        # prob_id = 'Proband02'
        # cond = '101'
        ###########################################################
        for clip in range(1, int(N_CLIPS + 1)):
            scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
            start_pos = (clip - 1) * CLIP_SIZE
            end_pos = clip * CLIP_SIZE
            print(cond + '-' + prob_id + '-clip' + str(clip))
            for idx in range(start_pos, end_pos):
                if idx % 100 == 0:
                    print("reading in frame " + str(idx) + "," + str(idx + 1))
                pre_path = scr_path + str(idx) + '.jpg'
                if idx == end_pos - 1 and clip != N_CLIPS:
                    print('end of clip-' + str(clip))
                    print('reading in ' + str((clip) * CLIP_SIZE) + '.jpg')
                    scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip + 1) + '/'
                    next_path = scr_path + str(clip * CLIP_SIZE) + '.jpg'
                elif idx == end_pos - 1 and clip == N_CLIPS:
                    print('done ' + cond + '-' + prob_id)
                    continue
                else:
                    next_path = scr_path + str(idx + 1) + '.jpg'
            #    print(os.path.exists(pre_path))
             #   print(os.path.exists(next_path))
                if not ( os.path.exists(pre_path) and os.path.exists(next_path)):
                    continue
                pre_frame = cv2.imread(pre_path).astype(np.float32)
                next_frame = cv2.imread(next_path).astype(np.float32)
                diff = np.subtract(next_frame, pre_frame)
                mean_fr = np.add(next_frame / 2.0, pre_frame / 2.0)
                re = np.true_divide(diff, mean_fr, dtype=np.float32)
                re[re == np.inf] = 0
                re = np.nan_to_num(re)
                li.append(re)
        mean = np.mean(li, axis=(0,1,2))
        std = np.std(li, axis=(0,1,2))
        print(mean.shape)
        col.append((mean, std))
    return cond, col


def cal_meanStd_label(label_paths, data_len=8):
    sgn_li = []
    skip_step = 256.0 / 30.0
    for label_path in label_paths:
        file_label = []
        #####remote####################################
        print(label_path)
        print(os.path.exists(label_path))
        path = label_path.split('/')
        prob_id = path[4]
        cond = path[5].split('_')[0]
        print(cond + '-' + prob_id)
        ######local##############################
        #cond = '101'
        #########################################
        labels = cvt_sensorSgn(label_path, skip_step)
        for idx in range(len(labels) - 1):
            val = float(labels[idx + 1] - labels[idx])
          #  if val > 0.2:
          #      val = 1
          #  elif val < -0.2:
          #      val = -1
          #  else:
          #      val = 0
            file_label.append(val)
        mean = np.mean(file_label)
        std = np.std(file_label)
        sgn_li.append((mean, std))
    return cond, sgn_li


def rescale_label(val, mean, std, model='classification'):
    val = val - mean
    val = val / std
    if val > 10:
        val = 10
    if val < -10:
        val = -10
#    if model == 'classification':
#        if val > 0.4:
#            val = [0, 0, 0, 1]
#        elif val < -0.4:
#            val = [1, 0, 0, 0]
#        elif 0 < val < 0.4:
#            val = [0, 0, 1, 0]
#        else:
#            val = [0, 1, 0, 0]
#    else:
#        if val > 0.4:
#            val = 1
#        elif val < -0.4:
#            val = -1
#        elif 0 < val < 0.4:
#            val = 0.2
#        else:
#            val = -0.2
    return val


def rescale_frame(img, mean=0, dev=1.0):
    mean = mean.reshape((1, 1, 3))
    dev = dev.reshape((1, 1, 3))
    img = img - mean
    img = np.true_divide(img, dev)
    return img


def clip_dframe(re, mean=0, dev=1.0):
    mean = mean.reshape((1, 1, 3))
    dev = dev.reshape((1, 1, 3))
    re = re - mean
    re = np.true_divide(re, dev)
    re[np.where(re>3)] = 3
    re[np.where(re<-3)] = -3
    return re


def get_meanstd(video_path, mode='video'):
    if mode == 'video':
        with open('MeanStddev.pickle', 'rb') as file:
            mean_std = pickle.load(file)
    elif mode == 'diff':
        with open('DiffFrameMeanStddev.pickle', 'rb') as file:
            mean_std = pickle.load(file)
    else:
        with open('LabelMeanStddev.pickle', 'rb') as file:
            mean_std = pickle.load(file)
    ########remote part########################################
    path = video_path.split('/')
    u_id = int(path[4][-1])
    t_id = int(path[4][-2])
    prob_id = t_id * 10 + u_id - 1
    cond = path[5].split('_')[0]
    #########local part##########################################
    # cond = '101'
    # prob_id = 1
    #############################################################
    #print('mean&std of '+mode+': '+cond + ' ' + str(prob_id))
    mean, dev = mean_std[cond][prob_id]
    # print(mean)
    # print(dev)
    file.close()
    return mean, dev


def cvt_sensorSgn(label_path, skip_step, data_len=8):
    labels = []
    binFile = open(label_path, 'rb')
    idx = 0
    try:
        while True:
            pos = math.floor(idx * skip_step)
            binFile.seek(pos * data_len)
            sgn = binFile.read(data_len)
            d_sgn = struct.unpack("d", sgn)[0]
            labels.append(d_sgn)
            idx += 1
    except Exception:
        pass
    binFile.close()
    return labels


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data)
    return y


def cheby2_bandpass_filter(data, rs, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby2(order, rs, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


#if __name__ == '__main__':
    #######remote&whole#######mean&std file####################################################
    #dict = {}
    #for cond in ['lighting','movement']:
    #   if cond == 'lighting':
    #       n = 6
    #   else:
    #       n = 4
    #   for i in range(n):
    #       vd, _ = create_file_paths(range(1, 27), cond=cond, cond_typ=i)
    #       con, col = cal_meanStd_video(vd)
    #       dict[con] = col
    #with open('MeanStddev.pickle', 'wb') as f:
    #   pickle.dump(dict, f)
    #f.close()
    ##########remote&part#####mean&std file###############################################################
    # dict = {}
    # vd, _ = create_file_paths(range(1, 27))
    # con, col = cal_meanStd_video(vd)
    # dict[con] = col
    # with open('MeanStddev.pickle', 'wb') as f:
    #     pickle.dump(dict, f)
    # f.close()
    # con, col = create_meanStd_file(VIDEO_PATHS)
    #########################check whether can get mean&std##############################################
    # for cond in ['lighting', 'movement']:
    #     if cond == 'lighting':
    #         n = 6
    #     else:
    #         n = 4
    #     for i in range(n):
    #         vd, lb = create_file_paths(range(9, 12), cond=cond, cond_typ=i)
    #         for v in vd:
    #             get_meanstd(v)
    #######remote&whole#######diff-frame mean&std file####################################################
    #dict = {}
    #for cond in ['lighting','movement']:
    #   if cond == 'lighting':
    #       n = 6
    #   else:
    #       n = 4
    #   for i in range(n):
    #       vd, _ = create_file_paths(range(1, 27), cond=cond, cond_typ=i)
    #       con, col = cal_meanStd_video(vd)
    #       dict[con] = col
    #with open('DiffFrameMeanStddev.pickle', 'wb') as f:
    #   pickle.dump(dict, f)
    #f.close()
    #######remote&part#######diff-frame mean&std file####################################################
    # dict = {}
    # vd, _ = create_file_paths(range(1, 27))
    # con, col = cal_meanStd_vdiff(vd)
    # dict[con] = col
    # with open('DiffFrameMeanStddev.pickle', 'wb') as f:
    #    pickle.dump(dict, f)
    # f.close()
    #########remote mean&std labels#####################################################################
    # dict = {}
    # for cond in ['lighting', 'movement']:
    #     if cond == 'lighting':
    #         n = 6
    #     else:
    #         n = 4
    #     for i in range(n):
    #         _, lb = create_file_paths(range(1, 27), cond=cond, cond_typ=i)
    #         con, col = cal_meanStd_label(lb)
    #         dict[con] = col
    # with open('LabelMeanStddev.pickle', 'wb') as f:
    #     pickle.dump(dict, f)
    # f.close()
    #########local mean&std labels#####################################################################
    # dict = {}
    # con = ''
    # col = []
    # con, col = cal_meanStd_label(LABEL_PATHS)
    # dict[con] = col
    # with open('LabelMeanStddev.pickle', 'wb') as f:
    #     pickle.dump(dict, f)
    # f.close()
#############################check generated labels###########################################
# li = cvt_labels(1,8)
# for i in li:
#     print(i)
# print(len(li))
