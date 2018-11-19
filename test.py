import numpy as np
import cv2
import pickle

# img = cv2.imread('C:/Users/Iris/Desktop/1.jpg')
# a = np.std(img, axis=(0,1))
# z = np.true_divide(img, a)
# print(img.shape)
# print(a)
# print(z.shape)
with open('MeanStddev.pickle','rb') as file:
    re = pickle.load(file)
print(len(re['204']))
print(re['202'][1])
