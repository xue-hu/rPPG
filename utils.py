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

VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']


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
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
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
    return faces.astype(int)


def create_meanStd_file(video_paths):
    col = []
    for v_path in video_paths:
        print(os.path.exists(v_path))
        path = v_path.split('/')
        prob_id = path[4]
        cond = path[5].split('_')[0]
        print(cond)
        print(prob_id)
        capture = cv2.VideoCapture()
        capture.release()
        ##########WAIT TO CHANGE##############
        capture.open(v_path)
        ######################################
        if not capture.isOpened():
            return -1
        nframe = int(capture.get(7))
        mean = 0
        dev = 0
        for idx in range(nframe):
            if idx % 100 == 0:
                print(idx)
            rd, frame = capture.read()
            f_mean = np.mean(frame, axis=(0, 1))
            mean += np.true_divide(f_mean, nframe)
            f_dev = np.std(frame, axis=(0, 1)) ** 2
            dev += np.true_divide(f_dev, nframe)
            idx += 1
        stddev = np.sqrt(dev)
        capture.release()
        col.append((mean, stddev))
    return cond, col


def create_file_paths(probs, cond='lighting', cond_typ=0, sensor_sgn=1):
    src_path = '/Vitalcam_Dataset/07_Datenbank_Smarthome/Testaufnahmen/Proband'
    conditions = {'lighting': ['/101_natural_lighting', '/102_artificial_lighting',
                               '/103_abrupt_changing_lighting', '/104_dim_lighting_auto_exposure',
                               '/106_green_lighting', '/107_infrared_lighting'],
                  'movement': ['/201_shouldercheck', '/202_scale_movement', '/203_translation_movement',
                               '/204_writing']}
    video_name = '/Logitech HD Pro Webcam C920.avi'
    label_name = '/synced_Logitech HD Pro Webcam C920/'
    sgn_typ = ['1_EKG-AUX.bin', '5_Pleth.bin', '6_Pulse.bin']

    video_paths = []
    label_paths = []

    for i in probs:
        prob_id = str(i) if (i > 9) else ('0' + str(i))
        video_path = src_path + prob_id + conditions[cond][cond_typ] + video_name
        label_path = src_path + prob_id + conditions[cond][cond_typ] + label_name + sgn_typ[sensor_sgn]
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
    print(cond + ' ' + str(prob_id) + ':')
    mean, dev = mean_std[cond][prob_id]
    # print(mean)
    # print(dev)
    file.close()
    return mean.reshape((1, 1, 3)), dev.reshape((1, 1, 3))


def rescale_frame(img, mean=0, dev=1.0):
    img = img - mean #- np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))
    img = np.true_divide(img, dev)
    return img


def clip_dframe(img, mean=0, deviation=1.0):
    # img = img - np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))
    return img


def cvt_sensorSgn(label_path, skip_step, data_len=8):
    # src_path = '/Vitalcam_Dataset/07_Datenbank_Smarthome/Testaufnahmen/Proband'
    # sgn_typ = ['1_EKG-AUX.bin', '5_Pleth.bin', '6_Pulse.bin']
    # for i in range(n_probs):
    # prob_id = str(i) if (i > 9) else ('0' + str(i))
    # label_path = src_path + prob_id + '/Proband_' + prob_id + '_unisens/' + sgn_typ[1]
    # binFile = open(label_path,'rb')
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


#if __name__ == '__main__':
    #######################generate mean&std file####################################################
    # dict = {}
    # con = ''
    # col = []
    # for cond in ['lighting','movement']:
    #     if cond == 'lighting':
    #         n = 6
    #     else:
    #         n = 4
    #     for i in range(n):
    #         vd, lb = create_file_paths(range(1,27), cond=cond, cond_typ=i)
    #         con, col = create_meanStd_file(vd)
    #         dict[con] = col
    # with open('MeanStddev.pickle', 'wb') as f:
    #     pickle.dump(dict, f)
    # f.close()
    ###################################################################################################
    #############################check generated labels###########################################
    # li = cvt_labels(1,8)
    # for i in li:
    #     print(i)
    # print(len(li))
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
    ####################################################################################################