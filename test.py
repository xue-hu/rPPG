import numpy as np
import cv2
import pickle
import random
import scipy
from scipy import fftpack
from scipy.signal import butter, cheby2, lfilter
import struct
import imblearn
from imblearn.over_sampling import RandomOverSampler
#import matplotlib.pyplot as plt

LABEL_PATHS = 'D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin'
GT_PATHS = 'D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/6_Pulse.bin'


ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(X, y)


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




