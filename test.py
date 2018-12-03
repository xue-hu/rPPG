import numpy as np
import cv2
import pickle
import random
import scipy
from scipy import fftpack
from scipy.signal import butter, cheby2, lfilter
import struct
#import matplotlib.pyplot as plt

LABEL_PATHS = 'D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin'
GT_PATHS = 'D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/6_Pulse.bin'
############check calculated mean&std##########################################
# with open('LabelMeanStddev.pickle', 'rb') as f:
#     m = pickle.load(f)
# print(m['101'][0])
# print(len(m['203']))

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




############check label diff distribution##########################################
with open('Pleth.pickle', 'rb') as f:
    m = pickle.load(f)
f.close()
pos = []
neg = []
li = []
gt = []
for val, hr in m['13']:
    gt.append(hr)
    print(str(val) + ' - ' + str(hr))
    if val > 0.2:
        val = 1
    elif val < -0.2:
        val = -1
    elif val> -0.2 and val<0:
        val = -0.5
    else:
        val = 0.5
    if val > 0:
        pos.append(val)
    elif val < 0:
        neg.append(val)
    li.append(val)
fig, axs = plt.subplots(2, 2, tight_layout=True)
axs[0][0].hist(m['13'], bins=30)
axs[0][1].hist(li, bins=30)
axs[1][0].hist(pos, bins=30)
axs[1][1].hist(neg, bins=30)
plt.title("label difference distribution")
plt.xlabel('value')
plt.ylabel('occurance')
plt.show()
# a = np.array([1,0,0])
# print(a)



