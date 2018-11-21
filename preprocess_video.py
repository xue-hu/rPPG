#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import tensorflow as tf
import cv2
import utils
import numpy as np
import math

ECG_SAMPLE_RATE = 16.0
PLE_SAMPLE_RATE = 256.0
FRAME_RATE = 30.0
VIDEO_DUR = 120
VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']
LABEL_PATHS = ['D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin']


def create_video_clip(video_paths, width=256, height=256):
    for video_path in video_paths:
        ########REMOTE####################################################
        # print(v_path)
        # print(os.path.exists(v_path))
        # path = v_path.split('/')
        # prob_id = path[4]
        # cond = path[5].split('_')[0]
        #######LOCAL####################################################
        prob_id = '01'
        cond = 'lighting'
        ##################################################################
        print(cond)
        print(prob_id)
        if not os.path.exists('./processed_video/' + cond + '/' + prob_id + '/'):
            os.makedirs('./processed_video/' + cond + '/' + prob_id + '/')
        capture = cv2.VideoCapture()
        capture.open(video_path)
        if not capture.isOpened():
            return -1
        else:
            print("video opened. start to read in.....")
        frame_height = int(capture.get(4))
        nframe = int(capture.get(7))

        for idx in range(nframe):
            if idx % 100 == 0:
                print("reading in frame " + str(idx))
            rd, frame = capture.read()
            if not rd:
                return -1
            faces = utils.detect_face(frame)
            if len(faces) != 0:
                for (x, y, w, h) in faces:
                    h = min(int(1.6 * h), (frame_height - y))
                    frame = frame[y:y + h, x:x + w]
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(('./processed_video/' + cond + '/' + prob_id + '/' + str(idx) + '.jpg'), frame, params=100)
                    break
                    # cv2.imshow('frame', frame)
                    # cv2.waitKey(0)
        print('done:  ' + cond + '/' + prob_id + '/')
        capture.release()


if __name__ == '__main__':
    ##########batched labeled-samples######################
    for cond in ['lighting', 'movement']:
        if cond == 'lighting':
            n = 6
        else:
            n = 4
        for i in range(n):
            vd, _ = create_file_paths(range(1, 27), cond=cond, cond_typ=i)
            create_video_clip(vd)
    #create_video_clip(VIDEO_PATHS)
