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


class TwoStreamVgg(cnn_model.CnnModel):
    def construct_network(self,img_width, img_height):
        print("begin to construct two stream vgg......")
        
        with tf.name_scope('Static_Input'):
            self.input_img = tf.placeholder(dtype=tf.float32, name='input_img',
                                            shape=[self.batch_size,img_width, img_height, 3])
        with tf.name_scope('Dynamic_Input'):
            self.input_diff = tf.placeholder(dtype=tf.float32, name='input_diff',
                                             shape=[self.batch_size, img_width, img_height, 3])
            
        self.conv2d_tanh(self.input_diff, 32, 'conv1_1',lyr_idx=0)
        self.conv2d_tanh(self.d_conv1_1, 32, 'conv1_2')
        
        self.conv2d_tanh(self.input_img,32, 'conv1_1',lyr_idx=0, stream_name='s')
        self.conv2d_tanh(self.s_conv1_1, 32, 'conv1_2', stream_name='s')
        
        self.attention_layer(self.d_conv1_2, self.s_conv1_2, 'd_conv1_2', 's_conv1_2')
        self.avgpool(self.atten_conv1_2, 'pool1')
        self.avgpool(self.s_conv1_2, 'pool1', stream_name='s')
        self.dropout_layer(self.d_pool1)

        

        self.conv2d_tanh(self.d_pool1, 64, 'conv2_1')
        self.conv2d_tanh(self.d_conv2_1, 64, 'conv2_2')
        
        self.conv2d_tanh(self.s_pool1, 64, 'conv2_1', stream_name='s')
        self.conv2d_tanh(self.s_conv2_1, 64, 'conv2_2', stream_name='s')
        
        self.attention_layer(self.d_conv2_2, self.s_conv2_2, 'd_conv2_2', 's_conv2_2')
        self.avgpool(self.atten_conv2_2, 'pool2')
        self.dropout_layer(self.d_pool2)

        self.fully_connected_layer(self.d_pool2, 128, 'fc7')
        self.dropout_layer(self.fc7)

        self.fully_connected_layer(self.fc7, 1, 'reg_output', last_lyr=True)
        self.fully_connected_layer(self.fc7, N_CLASSES, 'class_output', last_lyr=True)
        print('param nums: '+ str(self.t))
        print("done.")  
        
    def get_data(self, video_paths, label_paths, gt_paths, clips,width=112, height=112, mode='train'):
        print("create generator....")
        batch_gen = process_data.get_batch(video_paths, label_paths, gt_paths, clips, self.batch_size,
                                           width=width, height=height, mode=mode)
        return batch_gen

if __name__ == '__main__':
    m = TwoStreamVgg(64)
    m.construct_network(112,112)