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

#video_paths=['a','b','c','d']
#label_paths=['A','B','C','D']
#clips = [1,2,3,4,5]
#paths = [(v_p, l_p, clip) for v_p, l_p in zip(video_paths, label_paths) for clip in clips]
##random.shuffle(paths)
#print(len(paths))
#for (v, l, c) in paths:
#    print(v+' '+l+' '+str(c))
a = [0,-1,0,1,2,3,4,5,6,7]
for i in range(-8,0,1):
    s = i - 2
    print(str(i)+' '+str(a[s:i]))











