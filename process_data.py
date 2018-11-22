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
N_FRAME = 3600
CLIP_SIZE = 720
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
        if not rd:
            return -1
        faces = utils.detect_face(frame)
        if len(faces) != 0:
            for (x, y, w, h) in faces:
                # Convert bounding box to two CvPoints
                # pt1 = (int(x), int(0.9*y))
                # pt2 = (int(x + w), int(y + 1.6*h))
                # cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 5, 8, 0)
                h = min(int(1.6 * h), (frame_height - y))
                frame = frame[y:y + h, x:x + w]
                #frame = utils.rescale_frame(frame, mean, dev)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)
                #cv2.imwrite(('./'+str(idx)+'.jpg'),frame)
                # frame = np.expand_dims(frame, 0)
                yield frame
    capture.release()

#########version 1: piplining#########################################
# def nor_diff_face(video_path, width=112, height=112):
#     capture = cv2.VideoCapture()
#     capture.release()
#     capture.open(video_path)
#     if not capture.isOpened():
#         return -1
#     # frame_width = int(capture.get(3))
#     else:
#         print("video opened. start to read in.....")
#     mean, dev = utils.get_meanstd(video_path)
#     frame_height = int(capture.get(4))
#     nframe = int(capture.get(7))
#
#     for idx in range(nframe - 1):
#         print("reading in frame " + str(idx) + "," + str(idx + 1))
#         if idx == 0:
#             rd, pre_frame = capture.read()
#             if not rd:
#                 return -1
#         else:
#             pre_frame = next_frame
#
#         rd, next_frame = capture.read()
#         if not rd:
#             return -1
#         pre_faces = utils.detect_face(pre_frame)
#         next_faces = utils.detect_face(next_frame)
#         if idx%100 == 0:
#             print(pre_faces.shape)
#             print(pre_faces)
#         if len(pre_faces) != 0 and len(next_faces) != 0:
#             for (x1, y1, w1, h1), (x2, y2, w2, h2) in zip(pre_faces, next_faces):
#                 h1 = min(int(1.6 * h1), (frame_height - y1))
#                 h2 = min(int(1.6 * h2), (frame_height - y2))
#                 p_frame = pre_frame[y1:y1 + h1, x1:x1 + w1]
#                 n_frame = next_frame[y2:y2 + h2, x2:x2 + w2]
#                 # cv2.imshow("pre", p_frame)
#                 # cv2.imshow("next", n_frame)
#                 # cv2.waitKey(0)
#                 ###################### wait to check ###############################################
#                 pre_face = cv2.resize(p_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
#                 next_face = cv2.resize(n_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
#                 # cv2.imwrite(('./precessed_data'+str(idx)+'.jpg'),frame)
#                 diff = np.subtract(next_face, pre_face)
#                 mean_fr = np.add(next_face / 2.0, pre_face / 2.0)
#                 re = np.true_divide(diff, mean_fr, dtype=np.float32)
#                 re[re == np.inf] = 0
#                 re = np.nan_to_num(re)
#                 ########### wait to implement ########################################################
#                 re = utils.clip_dframe(re, deviation=3.0)
#                 ########################################################################################
#                 # cv2.imshow("diff", diff)
#                 #cv2.imshow("mean", mean.astype(np.uint8))
#                 #cv2.imshow("re", re)
#                 #cv2.waitKey(0)
#                 yield pre_face, re
#     capture.release()
######################################################################


#########version 2: preprocess#########################################
def nor_diff_clip(video_path, clip=1, width=112, height=112):
    ###########remote##########################################
    print(video_path)
    print(os.path.exists(video_path))
    path = video_path.split('/')
    prob_id = path[4]
    cond = path[5].split('_')[0]
    ###########local###########################################
    # prob_id = 'Proband02'
    # cond = '101'
    ###########################################################
    scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
    start_pos = (clip - 1) * CLIP_SIZE
    end_pos = clip * CLIP_SIZE - 1
    mean, dev = utils.get_meanstd(video_path)
    print(cond+'-'+prob_id+'-clip'+str(clip))
    for idx in range(start_pos, end_pos):
        print("reading in frame " + str(idx) + "," + str(idx + 1))
        pre_path = scr_path + str(idx)+'.jpg'
        next_path = scr_path + str(idx + 1) + '.jpg'
        pre_frame = cv2.imread(pre_path).astype(np.float32)
        next_frame = cv2.imread(next_path).astype(np.float32)
        #pre_frame = utils.rscale_frame(mean, dev)
        #next_frame = utils.rscale_frame(mean, dev)
        #pre_frame = cv2.resize(pre_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        #next_frame = cv2.resize(next_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        diff = np.subtract(next_frame, pre_frame)
        mean_fr = np.add(next_frame / 2.0, pre_frame / 2.0)
        re = np.true_divide(diff, mean_fr, dtype=np.float32)
        re[re == np.inf] = 0
        re = np.nan_to_num(re)
        ########### wait to implement ########################################################
        re = utils.clip_dframe(re, deviation=3.0)
        ########################################################################################
        # cv2.imshow("diff", diff)
        #cv2.imshow("mean", mean.astype(np.uint8))
        #cv2.imshow("re", re)
        #cv2.waitKey(0)
        yield pre_frame, re
    capture.release()
######################################################################


def get_sample(video_path, label_path, clip=1, width=112, height=112):
    diff_iterator = nor_diff_clip(video_path, clip=clip, width=width, height=height)
    skip_step = PLE_SAMPLE_RATE / FRAME_RATE
    labels = utils.cvt_sensorSgn(label_path, skip_step)
    start_pos = (clip - 1) * CLIP_SIZE
    end_pos = clip * CLIP_SIZE - 1

    for idx in range(start_pos, end_pos):
        frame, diff = next(diff_iterator)
        label = float(labels[idx])
        yield (frame, diff, label)


def get_batch(video_paths, label_paths, clips, batch_size, width=112, height=112):
    frame_batch = []
    diff_batch = []
    label_batch = []
    for i in clips:
        for video_path, label_path in zip(video_paths, label_paths):
            iterator = get_sample(video_path, label_path, i, width=width, height=height)
            try:
                while True:
                    while len(frame_batch) < batch_size:
                        frame, diff, label = next(iterator)
                        frame_batch.append(frame)
                        diff_batch.append(diff)
                        label_batch.append(label)
                    yield frame_batch, diff_batch, label_batch
                    # print('done one batch.')
                    frame_batch = []
                    diff_batch = []
                    label_batch = []
            except StopIteration:
                continue



if __name__ == '__main__':
    ##########batched labeled-samples######################
    #v_paths, l_paths = utils.create_file_paths([2,3])
    #batch_gen = get_batch(v_paths, l_paths, [1,2], 500)
    #######################################################
    batch_gen = get_batch(VIDEO_PATHS, LABEL_PATHS, [1,2], 500)
    try:
        while True:
            frames, diffs, labels = next(batch_gen)
            for (frame, diff, label) in zip(frames, diffs, labels):
                # cv2.imwrite(('frame'+ str(idx) + '.jpg'), frame)
                # cv2.imwrite(('diff'+ str(idx) + '.jpg'), diff)
                cv2.imshow('face', frame.astype(np.uint8))
                # cv2.imshow('diff', diff)
                print(label)
                cv2.waitKey(0)
                break
    except StopIteration:
        pass

    ######################################################
