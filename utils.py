__author__ = 'Iris'

import os
import urllib
import PIL
from PIL import ImageOps,Image
import numpy as np
import cv2
import struct
import math


def download(down_link, file_path, expected_bytes):
    if os.path.exists(file_path):
        print("model exists. No need to dwonload.")
        return
    print("downloading......")
    model_file,_ = urllib.request.urlretrieve(down_link, file_path)
    assert os.stat(model_file).st_size == expected_bytes
    print("successfully downloaded.")


def rescale_image(img, mean=0, deviation=1.0):
    return img


def detect_face(image):
    min_size = (20,20)
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    temp = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    cv2.equalizeHist(temp, temp)

    faces = faceCascade.detectMultiScale(
            temp,
            haar_scale, min_neighbors, haar_flags, min_size
        )
    # if faces.size != 0 :
    #     for (x, y, w, h) in faces:
    #         # Convert bounding box to two CvPoints
    #         pt1 = (int(x), int(0.9*y))
    #         pt2 = (int(x + w), int(y + 1.8*h))
    #         cv2.rectangle(image, pt1, pt2, (255, 0, 0), 5, 8, 0)
    #         cv2.namedWindow('face',0)
    #         cv2.waitKey(0)
    #         cv2.imshow('face',image)
    return faces.astype(int)


def cvt_labels(label_path,skip_step, data_len=8 ):
    # src_path = '/Vitalcam_Dataset/07_Datenbank_Smarthome/Testaufnahmen/Proband'
    # sgn_typ = ['1_EKG-AUX.bin', '5_Pleth.bin', '6_Pulse.bin']
    #for i in range(n_probs):
        # prob_id = str(i) if (i > 9) else ('0' + str(i))
        # label_path = src_path + prob_id + '/Proband_' + prob_id + '_unisens/' + sgn_typ[1]
        # binFile = open(label_path,'rb')
    label_path = 'D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin'
    labels = []
    binFile = open(label_path, 'rb')
    idx=0
    try:
        while True:
            pos = math.floor(idx*skip_step)
            binFile.seek(pos*data_len)
            sgn = binFile.read(data_len)
            d_sgn = struct.unpack("d",sgn)[0]
            labels.append(d_sgn)
            idx+=1
    except Exception:
        pass
    binFile.close()
    return labels



def create_file_paths(probs):
    src_path = '/Vitalcam_Dataset/07_Datenbank_Smarthome/Testaufnahmen/Proband'
    conditions = {'lighting': ['/101_natural_lighting', '/102_artificial_lighting',
                  '/103_abrupt_changing_lighting', '/104_dim_lighting_auto_exposure',
                  '/106_green_lighting', '/107_infrared_lighting'],
                  'movement': ['/202_scale_movement', '/203_translation_movement',
                                '/204_writing', '/201_shouldercheck']}
    video_name = '/synced_Logitech HD Pro Webcam C920.avi'

    sgn_typ = ['1_EKG-AUX.bin', '5_Pleth.bin', '6_Pulse.bin']

    video_paths = []
    label_paths = []

    for i in probs:
        prob_id = str(i) if(i > 9) else ('0'+str(i))
        video_path = src_path+prob_id+conditions['lighting'][0]+video_name
        label_path = src_path+prob_id+'/Proband_'+prob_id+'_unisens/'+sgn_typ[1]
        video_paths.append(video_path)
        label_paths.append(label_path)
    # print(video_paths)
    # print(label_paths)
    return video_paths, label_paths



if __name__ == '__main__':
    #create_file_paths([1,10])
    li = cvt_labels(1,8)
    for i in li:
        print(i)
    print(len(li))









