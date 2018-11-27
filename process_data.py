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
from scipy.signal import butter, cheby2, lfilter
#import matplotlib.pyplot as plt

ECG_SAMPLE_RATE = 16.0
PLE_SAMPLE_RATE = 256.0
FRAME_RATE = 30.0
VIDEO_DUR = 120
N_FRAME = 3600
N_CLIPS = 5
CLIP_SIZE = N_FRAME / N_CLIPS
VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']
LABEL_PATHS = ['D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin']
GT_PATHS = ['D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/6_Pulse.bin']


def cvt_hr(labels, duration, fs, lowcut, highcut, order):
    N = len(labels)
    t = np.linspace(0, duration, N)
    # plt.figure(1)
    # plt.plot(t, labels)
    # plt.title("Unfiltered PPG data")
    # plt.xlabel('Time[sec]')
    # plt.ylabel('PPG data')
    # plt.show()

    y = utils.butter_bandpass_filter(labels, lowcut, highcut, fs, order)
    # y = cheby2_bandpass_filter(labels, 20, lowcut, highcut, fs, order=4)
    # y = labels
    # plt.figure(2)
    # plt.plot(t, labels, color ='crimson', label='data')
    # plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    # plt.xlabel('Time [sec]')
    # plt.ylabel('PPG data')
    # plt.title("Bandpass Filtered data for obtaining Heart Rate")
    # plt.grid()
    # plt.legend()
    #
    # plt.subplots_adjust(hspace=0.35)
    # plt.show()

    # periodogram
    FFT2 = abs(scipy.fft(y, N))
    f2 = 20 * scipy.log10(FFT2)
    f2 = f2[range(int(N / 2))]  # remove mirrored part of FFT
    freqs2 = scipy.fftpack.fftfreq(N, t[1] - t[0])
    freqs2 = freqs2[range(int(N / 2))]  # remove mirrored part of FFT

    d = np.argmax(f2)

    # Plotting periodogram
    # x1 = freqs2[d]
    # y1 = max(f2)
    # plt.figure(3)
    # plt.subplot(2,1,1)
    # plt.plot(freqs2, f2,color='darkmagenta')
    # plt.ylabel("PSD")
    # plt.title('Periodogram for Heart Rate detection')
    # plt.grid()
    # plt.subplot(2,1,2)
    # plt.plot(freqs2,f2,color='turquoise')
    # plt.xlim((0,10))
    # plt.ylim((0,y1+20))
    # plt.text(x1,y1,'*Peak corresponding to Maximum PSD')
    # plt.xlabel('Frequency(Hz)')
    # plt.ylabel('PSD')
    # plt.grid()
    # plt.show()

    # print('Maximum PSD:' , max(f2))
    # print("The frequency associated with maximum PSD is", freqs2[d], "Hz")

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
                d_sgn = struct.unpack("d", sgn)[0] #- 390.04378353
                labels.append(d_sgn)
                n += 1
            re = cvt_hr(labels, duration, fs, lowcut, highcut, order)
            hr.append(re)
        except Exception:
            pass
    binFile.close()
    return hr


def get_hr(labels, batch_size, duration, fs, lowcut=0.7, highcut=2.5, order=6):
    N = int(fs * duration)
    hr = []
    for idx in range((-batch_size), 0, 1):
        start_pos = idx - N
        end_pos = idx
        print('ppg fraction:'+str(start_pos)+'--'+str(end_pos))
        ppg = labels[start_pos:end_pos]
        #print(str(idx + batch_size) + '-' + len(ppg))
        hr.append(cvt_hr(ppg, duration, fs, lowcut, highcut, order))
    return hr


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
                # frame = utils.rescale_frame(frame, mean, dev)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)
                # cv2.imwrite(('./'+str(idx)+'.jpg'),frame)
                # frame = np.expand_dims(frame, 0)
                yield frame
    capture.release()


def nor_diff_face(video_path, width=112, height=112):
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
    for clip in range(1, int(N_CLIPS+1)):
        scr_path = './processed_video/' + cond + '/' + prob_id + '/' + str(clip) + '/'
        start_pos = (clip - 1) * CLIP_SIZE
        end_pos = clip * CLIP_SIZE
        mean, dev = utils.get_meanstd(video_path)
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
                raise StopIteration
            else:
                next_path = scr_path + str(idx + 1) + '.jpg'
            pre_frame = cv2.imread(pre_path).astype(np.float32)
            next_frame = cv2.imread(next_path).astype(np.float32)
            pre_frame = utils.rescale_frame(pre_frame, mean, dev)
            next_frame = utils.rescale_frame(next_frame, mean, dev)
            # pre_frame = cv2.resize(pre_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            # next_frame = cv2.resize(next_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            diff = np.subtract(next_frame, pre_frame)
            mean_fr = np.add(next_frame / 2.0, pre_frame / 2.0)
            re = np.true_divide(diff, mean_fr, dtype=np.float32)
            re[re == np.inf] = 0
            re = np.nan_to_num(re)
            ########### wait to implement ########################################################
            re = utils.clip_dframe(re, deviation=3.0)
            ########################################################################################
            # cv2.imshow("diff", diff)
            # cv2.imshow("mean", mean.astype(np.uint8))
            # cv2.imshow("re", re)
            # cv2.waitKey(0)
            yield pre_frame, re


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
    print(cond + '-' + prob_id + '-clip' + str(clip))
    for idx in range(start_pos, end_pos):
        if idx % 100 == 0:
            print("reading in frame " + str(idx) + "," + str(idx + 1))
        pre_path = scr_path + str(idx) + '.jpg'
        next_path = scr_path + str(idx + 1) + '.jpg'
        pre_frame = cv2.imread(pre_path).astype(np.float32)
        next_frame = cv2.imread(next_path).astype(np.float32)
        pre_frame = utils.rescale_frame(pre_frame, mean, dev)
        next_frame = utils.rescale_frame(next_frame, mean, dev)
        # pre_frame = cv2.resize(pre_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # next_frame = cv2.resize(next_frame, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        diff = np.subtract(next_frame, pre_frame)
        mean_fr = np.add(next_frame / 2.0, pre_frame / 2.0)
        re = np.true_divide(diff, mean_fr, dtype=np.float32)
        re[re == np.inf] = 0
        re = np.nan_to_num(re)
        ########### wait to implement ########################################################
        re = utils.clip_dframe(re, deviation=3.0)
        ########################################################################################
        # cv2.imshow("diff", diff)
        # cv2.imshow("mean", mean.astype(np.uint8))
        # cv2.imshow("re", re)
        # cv2.waitKey(0)
        yield pre_frame, re


######################################################################


def get_sample(video_path, label_path, clip=1, width=112, height=112, mode='train'):
    if mode == 'train':
        diff_iterator = nor_diff_clip(video_path, clip=clip, width=width, height=height)
        skip_step = PLE_SAMPLE_RATE / FRAME_RATE
        labels = utils.cvt_sensorSgn(label_path, skip_step)
        start_pos = (clip - 1) * CLIP_SIZE
        end_pos = clip * CLIP_SIZE - 1
        for idx in range(start_pos, end_pos):
            frame, diff = next(diff_iterator)
            label = float(labels[idx])
            yield (frame, diff, label)
    else:
        diff_iterator = nor_diff_face(video_path, width=width, height=height)
        skip_step = ECG_SAMPLE_RATE / FRAME_RATE
        labels = utils.cvt_sensorSgn(label_path, skip_step)
        for idx in range(N_FRAME - 1):
            frame, diff = next(diff_iterator)
            label = float(labels[idx])
            yield (frame, diff, label)


def get_batch(video_paths, label_paths, clips, batch_size, width=112, height=112, mode='train'):
    frame_batch = []
    diff_batch = []
    label_batch = []
    random.shuffle(clips)
    paths = list(zip(video_paths, label_paths))
    if mode == 'train':
        for clip in clips:
            random.shuffle(paths)
            for (video_path, label_path) in paths:
                iterator = get_sample(video_path, label_path, clip=clip, width=width, height=height, mode=mode)
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
    else:
        for (video_path, label_path) in zip(video_paths, label_paths):
            iterator = get_sample(video_path, label_path, width=width, height=height, mode=mode)
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
    # train_v_paths, train_l_paths = utils.create_file_paths([2,3])
    # train_gen = get_batch(train_v_paths, train_l_paths, [1, 2], 500)
    # test_v_paths, test_l_paths = utils.create_file_paths([3], sensor_sgn=0)
    # test_gen = get_batch(test_v_paths, test_l_paths, [1,2], 500, mode='test')

    #######################################################
    # train_gen = get_batch(VIDEO_PATHS, LABEL_PATHS, [1, 2], 500)
    # test_gen = get_batch(VIDEO_PATHS, GT_PATHS, [1, 2], 500, mode='test')
    # idx = 0
    # print('<<<<<<<<<train gen>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # try:
    #     while True:
    #         frames, diffs, labels = next(train_gen)
    #         print('batch-'+str(idx))
    #         idx += 1
    #         for (frame, diff, label) in zip(frames, diffs, labels):
            #     # cv2.imwrite(('frame'+ str(idx) + '.jpg'), frame)
            #     # cv2.imwrite(('diff'+ str(idx) + '.jpg'), diff)
            #     cv2.imshow('face', frame.astype(np.uint8))
            #     # cv2.imshow('diff', diff)
            #     print(label)
            #     cv2.waitKey(0)
    #             break
    # except StopIteration:
    #     pass
    # print('<<<<<<<<<test gen>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
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

    ############cvt ppg to hr##########################################
    #gt_paths = utils.create_file_paths([2], sensor_sgn=0)
    #labels_paths = utils.create_file_paths([2], sensor_sgn=1)
    binFile = open(GT_PATHS[0], 'rb')
    data_len = 8
    duration = 30
    skip_step = 16
    idx = 0
    gt = []
    for idx in range(duration, 120):
        pos = 16 * idx
        binFile.seek(pos * data_len)
        sgn = binFile.read(data_len)
        d_sgn = struct.unpack("d", sgn)[0]
        gt.append(d_sgn)
    binFile.close()
    hr = test_hr(LABEL_PATHS[0], 30, 30)
    for rate, g in zip(hr, gt):
        print(str(round(rate)) + '  ' + str(g))
###########################################################################
