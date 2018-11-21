import numpy as np
import cv2
import pickle
with open('MeanStddev.pickle', 'rb') as f:
    m = pickle.load(f)
print(m['101'][0])
print(len(m['101']))
