#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import tensorflow as tf
import utils
import scipy.io
import numpy as np
import cv2
import cnn_model
import process_data

# VGG-19 parameters file
N_CLASSES = 2
# VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
# VGG_FILENAME = 'imagenet-vgg-verydeep-19.mat'
# EXPECTED_BYTES = 534904783
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/vgg-face.mat'
VGG_FILENAME = 'vgg-face.mat'
EXPECTED_BYTES = 1086058494


class FaceVgg(cnn_model.CnnModel):
    def construct_network(self,img_width, img_height):
        print("begin to construct two stream vgg......")
        
        with tf.name_scope('Input'):
            self.input_img = tf.placeholder(dtype=tf.float32, name='input_img',
                                            shape=[self.batch_size,img_width, img_height, 3])
            
        self.conv2d_relu(self.input_img, 0, 'conv1_1')
        self.conv2d_relu(self.conv1_1, 2, 'conv1_2')        
        self.avgpool(self.conv1_2, 'pool1', stream_name='s')
        #self.dropout_layer(self.s_pool1)

        self.conv2d_relu(self.s_pool1, 5, 'conv2_1')
        self.conv2d_relu(self.conv2_1, 7, 'conv2_2')
        self.avgpool(self.conv2_2, 'pool2', stream_name='s')
        #self.dropout_layer(self.s_pool2)

        self.conv2d_relu(self.s_pool2, 10, 'conv3_1')
        self.conv2d_relu(self.conv3_1, 12, 'conv3_2')
        self.conv2d_relu(self.conv3_2, 14, 'conv3_3')
        self.avgpool(self.conv3_3, 'pool3', stream_name='s')
        #self.dropout_layer(self.s_pool3)
        
        self.conv2d_relu(self.s_pool3, 17, 'conv4_1')
        self.conv2d_relu(self.conv4_1, 19, 'conv4_2')
        self.conv2d_relu(self.conv4_2, 21, 'conv4_3')
        self.avgpool(self.conv4_3, 'pool4', stream_name='s')
        #self.dropout_layer(self.s_pool4)
        
        self.conv2d_relu(self.s_pool4, 24, 'conv5_1')
        self.conv2d_relu(self.conv5_1, 26, 'conv5_2')
        self.conv2d_relu(self.conv5_2, 28, 'conv5_3')
        #self.avgpool(self.conv5_3, 'pool5', stream_name='s')
        #self.dropout_layer(self.s_pool5)
        
        self.conv2d_relu(self.conv5_3, 31, 'fc6')
        self.conv2d_relu(self.fc6, 33, 'fc7')
        
    def get_data(self, video_paths, label_paths, gt_paths, clips,window_size,width=112, height=112):
        print("create generator....")
        batch_gen = process_data.get_seq_batch(video_paths, label_paths, gt_paths, self.batch_size,window_size,width=width, height=height)    
        return batch_gen
        
if __name__ == '__main__':
    m = FaceVgg(64)
    m.construct_network(112,112)