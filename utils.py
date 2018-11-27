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

VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']
LABEL_MEAN = 390.04378353 
LABEL_STD = 148.0124269

def download(down_link, file_path, expected_bytes):
    if os.path.exists(file_path):
        print("model exists. No need to dwonload.")
        return
    print("downloading......")
    model_file, _ = urllib.request.urlretrieve(down_link, file_path)
    assert os.stat(model_file).st_size == expected_bytes
    print("successfully downloaded.")


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
        print(v_path)
        print(os.path.exists(v_path))
        path = v_path.split('/')
        prob_id = path[4]
        cond = path[5].split('_')[0]
        print(cond + '-' + prob_id)
        capture = cv2.VideoCapture()
        capture.release()
        ##########WAIT TO CHANGE##############
        capture.open(v_path)
        ######################################
        if not capture.isOpened():
            return -1
        nframe = int(capture.get(7))
        frame_height = int(capture.get(4))
        mean = 0.0
        dev = 0.0
        for idx in range(nframe):
            if idx % 100 == 0:
                print(idx)
            rd, frame = capture.read()
            faces = detect_face(frame)
            #  print(faces)
            if len(faces) != 0:
                for (x, y, w, h) in faces:
                    h = min(int(1.6 * h), (frame_height - y))
                    frame = frame[y:y + h, x:x + w]
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                    f_mean = np.mean(frame, axis=(0, 1))
                    mean += np.true_divide(f_mean, nframe)
                    f_dev = np.std(frame, axis=(0, 1)) ** 2
                    dev += np.true_divide(f_dev, nframe)
                    break
        stddev = np.sqrt(dev)
        capture.release()
        col.append((mean, stddev))
    return cond, col


def cal_meanStd_label(label_paths, data_len=8):
    sgn_li = []
    for label_path in label_paths:
        print(label_path)
        print(os.path.exists(label_path))
        path = label_path.split('/')
        prob_id = path[4]
        cond = path[5].split('_')[0]
        print(cond+'-'+prob_id)
        binFile = open(label_path, 'rb')
        flag = True
        idx = 0
        try:
            while True:
                pos = int(math.floor(idx * (256.0 / 30.0)))
                if flag:
                    sgn = binFile.read(pos * data_len)
                    d_sgn = struct.unpack("d", sgn)[0]
                    idx += 1
                    pos = math.floor(idx * (256.0 / 30.0))
                    sgn2 = binFile.read(data_len)
                    d_sgn2 = struct.unpack("d", sgn2)[0]
                    idx += 1
                    flag = False
                else:
                    d_sgn = d_sgn2
                    sgn2 = binFile.read(pos * data_len)
                    d_sgn2 = struct.unpack("d", sgn2)[0]
                    idx += 1
                re = (d_sgn2 - d_sgn)
                sgn_li.append(re)
        except Exception:
            binFile.close()
            #continue
    mean = np.mean(sgn_li)
    std = np.std(sgn_li)
    return mean, std


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


def get_meanstd(video_path):
    with open('MeanStddev.pickle', 'rb') as file:
        mean_std = pickle.load(file)
    ########remote part########################################
    path = video_path.split('/')
    u_id = int(path[4][-1])
    t_id = int(path[4][-2])
    prob_id = t_id * 10 + u_id
    cond = path[5].split('_')[0]
    #########local part##########################################
    # cond = '101'
    # prob_id = 0
    #############################################################
    #print(cond + ' ' + str(prob_id) + ':')
    mean, dev = mean_std[cond][prob_id]
    # print(mean)
    # print(dev)
    file.close()
    return mean.reshape((1, 1, 3)), dev.reshape((1, 1, 3))


def rescale_frame(img, mean=0, dev=1.0):
    img = img - mean  # - np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))
    #img = np.true_divide(img, dev)
    return img


def clip_dframe(img, mean=0, deviation=1.0):
    # img = img - np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))
    return img


def cvt_sensorSgn(label_path, skip_step, data_len=8):
    labels = []
    binFile = open(label_path, 'rb')
    idx = 0
    try:
        while True:
            pos = math.floor(idx * skip_step)
            binFile.seek(pos * data_len)
            sgn = binFile.read(data_len)
            d_sgn = struct.unpack("d", sgn)[0] - LABEL_MEAN
            #d_sgn = d_sgn / LABEL_STD
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



if __name__ == '__main__':
    #######remote&whole#######mean&std file####################################################
    # dict = {}
    # con = ''
    # col = []
    # for cond in ['lighting','movement']:
    #   if cond == 'lighting':
    #      n = 6
    # else:
    #    n = 4
    # for i in range(n):
    #    vd, lb = create_file_paths(range(1, 27), cond=cond, cond_typ=i)
    #   con, col = cal_meanStd_video(vd)
    #  dict[con] = col
    #    with open('MeanStddev.pickle', 'wb') as f:
    #       pickle.dump(dict, f)
    #  f.close()
    ##########remote&part#####mean&std file###############################################################
    # dict = {}
    # con = ''
    # col = []
    # vd, lb = create_file_paths(range(1, 27), cond='lighting', cond_typ=0)
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
    ###############mean&std labels#####################################################################
    l_paths = []
    for cond in ['lighting', 'movement']:
        if cond == 'lighting':
            n = 6
        else:
            n = 4
        for i in range(n):
            _, lb = create_file_paths(range(1, 27), cond=cond, cond_typ=i)
            l_paths += lb
    mean, dev = cal_meanStd_label(l_paths)
    print(mean)
    print(dev)
#############################check generated labels###########################################
# li = cvt_labels(1,8)
# for i in li:
#     print(i)
# print(len(li))
