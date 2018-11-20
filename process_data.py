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


def crop_resize_face(video_path, width=112, height=112):
    capture = cv2.VideoCapture()
    capture.release()
    capture.open(video_path)
    if not capture.isOpened():
        return -1
    else:
        print("video opened. start to read in.....")
    mean, dev = utils.get_meanstd(video_path)
    # frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    framerate = capture.get(5)
    nframe = int(capture.get(7))

    for idx in range(nframe - 1):
        print("reading in frame " + str(idx))
        rd, frame = capture.read()
        frame = utils.rescale_frame(frame, mean, dev)
        if not rd:
            return -1
        faces = utils.detect_face(frame)
        if faces.size != 0:
            for (x, y, w, h) in faces:
                # Convert bounding box to two CvPoints
                # pt1 = (int(x), int(0.9*y))
                # pt2 = (int(x + w), int(y + 1.6*h))
                # cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 5, 8, 0)
                h = min(int(1.6 * h), (frame_height - y))
                frame = frame[y:y + h, x:x + w]
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)
                #cv2.imwrite(('./'+str(idx)+'.jpg'),frame)
                # frame = np.expand_dims(frame, 0)
                yield frame
    capture.release()


def nor_diff_face(video_path, width=112, height=112):
    capture = cv2.VideoCapture()
    capture.release()
    capture.open(video_path)
    if not capture.isOpened():
        return -1
    # frame_width = int(capture.get(3))
    else:
        print("video opened. start to read in.....")
    mean, dev = utils.get_meanstd(video_path)
    frame_height = int(capture.get(4))
    nframe = int(capture.get(7))

    for idx in range(nframe - 1):
        print("reading in frame " + str(idx) + "," + str(idx + 1))
        if idx == 0:
            rd, pre_frame = capture.read()
            pre_frame = utils.rescale_frame(frame, mean, dev)
            if not rd:
                return -1
        else:
            pre_frame = next_frame

        rd, next_frame = capture.read()
        next_frame = utils.rescale_frame(frame, mean, dev)
        if not rd:
            return -1
        idx += 1
        pre_faces = utils.detect_face(pre_frame)
        next_faces = utils.detect_face(next_frame)
        if pre_faces.size != 0 and next_faces.size != 0:
            for (x1, y1, w1, h1), (x2, y2, w2, h2) in zip(pre_faces, next_faces):
                h1 = min(int(1.6 * h1), (frame_height - y1))
                h2 = min(int(1.6 * h2), (frame_height - y2))
                p_frame = pre_frame[y1:y1 + h1, x1:x1 + w1]
                n_frame = next_frame[y2:y2 + h2, x2:x2 + w2]
                ###################### wait to check ###############################################
                pre_face = cv2.resize(p_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                next_face = cv2.resize(n_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                # cv2.imwrite(('./precessed_data'+str(idx)+'.jpg'),frame)
                diff = np.subtract(next_face, pre_face)
                mean_fr = np.add(next_face / 2.0, pre_face / 2.0)
                re = np.true_divide(diff, mean_fr, dtype=np.float32)
                re[re == np.inf] = 0
                re = np.nan_to_num(re)
                ########### wait to implement ########################################################
                re = utils.clip_dframe(re, deviation=3.0)
                ########################################################################################
                # cv2.imshow("pre",pre_face)
                # cv2.imshow("next", next_face)
                # cv2.imshow("diff", diff)
                #cv2.imshow("mean", mean.astype(np.uint8))
                #cv2.imshow("re", re)
                #cv2.waitKey(0)
                yield pre_face, re
    capture.release()


def get_sample(diff_iterator, label_paths):
    # with open(label_paths, 'r') as f:
    #     lines = f.readlines()
    skip_step = PLE_SAMPLE_RATE / FRAME_RATE
    labels = utils.cvt_sensorSgn(label_paths, skip_step)
    idx = 0
    while idx < (FRAME_RATE * VIDEO_DUR):
        frame, diff = next(diff_iterator)
        label = float(labels[idx])
        # label = float(lines[math.floor(idx*skip_step)])
        idx += 1
        yield (frame, diff, label)


def get_batch(iterator, batch_size):
    while True:
        frame_batch = []
        diff_batch = []
        label_batch = []
        try:
            for i in range(batch_size):
                frame, diff, label = next(iterator)
                frame_batch.append(frame)
                diff_batch.append(diff)
                label_batch.append(label)
            yield frame_batch, diff_batch, label_batch
        except Exception:
            pass


if __name__ == '__main__':
    ##########batched labeled-samples######################
    # v_paths, l_paths = utils.create_file_paths([2,3])
    # for v_path, l_path in zip(v_paths, l_paths):
    #######################################################
    for v_path, l_path in zip(VIDEO_PATHS, LABEL_PATHS):
        print(v_path)
        diff_gen = nor_diff_face(v_path)
        sample_gen = get_sample(diff_gen, l_path)
        batch_gen = get_batch(sample_gen, 3)
        idx = 0
        while idx < 6:
            frames, diffs, labels = next(batch_gen)
            for frame, diff, label in zip(frames, diffs, labels):
                # cv2.imwrite(('frame'+ str(idx) + '.jpg'), frame)
                # cv2.imwrite(('diff'+ str(idx) + '.jpg'), diff)
                idx += 1
                # cv2.imshow('face', frame)
                # cv2.imshow('diff', diff)
                print(label)
                #cv2.waitKey(0)
    ######################################################
