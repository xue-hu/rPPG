import numpy as np
import cv2
import pandas as pd

# img = cv2.imread('C:/Users/Iris/Desktop/1.jpg')
# a = np.std(img, axis=(0,1))
# z = np.true_divide(img, a)
# print(img.shape)
# print(a)
# print(z.shape)
df = pd.read_csv('MeanStddev.csv')
print(df['101'][1])