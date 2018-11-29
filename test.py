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

# with open('LabelMeanStddev.pickle', 'rb') as f:
#     m = pickle.load(f)
# print(m['101'][0])
# print(len(m['203']))

# video_paths=['a','b','c','d']
# label_paths=['A','B','C','D']
# paths = list(zip(video_paths, label_paths))
#
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



def get_remote_label(label_paths):
    sgns = []
    skip_step = 256.0 / 30.0
    for label_path in label_paths:
        labels = utils.cvt_sensorSgn(label_path, skip_step)
        mean, std = utils.get_meanstd(label_path, mode='label')
        for idx in range(len(labels) - 1):
            val = float(labels[idx + 1] - labels[idx])
            val = val - mean
            val = val / std
            if val > 0.5:
                val = 1
            elif val < 0:
                val = -1
            else:
                val = 0
            sgns.append(val)
    return sgns

dict = {}
for cond in ['lighting', 'movement']:
    if cond == 'lighting':
        n = 6
    else:
        n = 4
    for i in range(n):
        _, lb = create_file_paths(range(1, 27), cond=cond, cond_typ=i)
        sgn = get_remote_label(lb)
        dict['label'] = sgn
with open('Label.pickle', 'wb') as f:
    pickle.dump(dict, f)
f.close()








