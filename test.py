import numpy as np
import cv2
import pickle
import random
import scipy
from scipy import fftpack
from scipy.signal import butter, cheby2, lfilter
import struct

LABEL_PATHS = 'D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin'
GT_PATHS = 'D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/6_Pulse.bin'

# with open('MeanStddev.pickle', 'rb') as f:
#     m = pickle.load(f)
# print(m['101'][0])
# print(len(m['101']))

video_paths=['a','b','c','d']
label_paths=['A','B','C','D']
paths = list(zip(video_paths, label_paths))

clips = [1,2,3,4,5]
random.shuffle(clips)
for clip in clips:
    random.shuffle(paths)
    for vp, lp in paths:
        print(vp + ' ' + lp + ' ' + str(clip))
#random.shuffle(paths)
# print(len(paths))
# for (v, l, c) in paths:
#    print(v+' '+l+' '+str(c))












