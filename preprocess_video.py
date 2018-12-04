#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import tensorflow as tf
import cv2
import utils
import numpy as np
import math
import pickle

ECG_SAMPLE_RATE = 16.0
PLE_SAMPLE_RATE = 256.0
N_FRAME = 3600
FRAME_RATE = 30.0
VIDEO_DUR = 120
N_CLIPS = 120
CLIP_SIZE = int(N_FRAME / N_CLIPS)
VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']
LABEL_PATHS = ['D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin']


def create_video_clip(video_paths, width=128, height=128):
    for video_path in video_paths:
        ########REMOTE####################################################
        print(video_path)
        print(os.path.exists(video_path))
        path = video_path.split('/')
        prob_id = path[4]
        cond = path[5].split('_')[0]
        #######LOCAL####################################################
        #prob_id = 'Proband02'
        #cond = '101'
        ##################################################################
        print(cond)
        print(prob_id)
        if cond == '101' or cond =='102' or cond == '103':
            print('<<<<<<<<<<<skip>>>>>>>>>>>')
            continue
        capture = cv2.VideoCapture()
        capture.open(video_path)
        if not capture.isOpened():
            return -1
        else:
            print("video opened. start to read in.....")
        frame_height = int(capture.get(4))
        frame_width = int(capture.get(3))
        nframe = int(capture.get(7))
        clip = 0
        for idx in range(nframe):
            if idx % CLIP_SIZE == 0:
                clip += 1
            if not os.path.exists('./processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'):
                os.makedirs('./processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/')
            if idx % 100 == 0:
                print("reading in frame " + str(idx))
            rd, frame = capture.read()
            if not rd:
                return -1
            faces = utils.detect_face(frame)
            if len(faces) != 0:
                for (x, y, w, h) in faces:
                    y = max(int(0.95 * y), 0)
                    h = min(int(1.7 * h), (frame_height - y))
                    x = max(int(0.98 * x), 0)
                    w = min(int(1.2 * w), (frame_width - x))
                    frame = frame[y:y + h, x:x + w]
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(
                        ('./processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/' + str(idx) + '.jpg'),
                        frame)
                    break
                    # cv2.imshow('frame', frame)
                    # cv2.waitKey(0)
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
    #########1.remote-prepro all videos######################
    # for cond in ['lighting', 'movement']:
    #     if cond == 'lighting':
    #         n = 6
    #     else:
    #         n = 4
    #     for i in range(n):
    #         vd, _ = utils.create_file_paths(range(1, 27), cond=cond, cond_typ=i)
    #         create_video_clip(vd)
    # create_video_clip(VIDEO_PATHS)
    #########2.remote-prepro part of videos######################
    # vd, _ = utils.create_file_paths([9, 10])
    # v_d, _ = utils.create_file_paths(range(12, 27 ))
    # vd += v_d
    # create_video_clip(vd)
    ##########3.local-prepro part of videos######################
    #create_video_clip(VIDEO_PATHS)
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
