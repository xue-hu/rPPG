#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import time
import tensorflow as tf
import utils
import scipy.io
import numpy as np
import cv2
import math
import cnn_model
import process_data

# VGG-19 parameters file
N_CLASSES = 2
FRAME_RATE = 30.0
# VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
# VGG_FILENAME = 'imagenet-vgg-verydeep-19.mat'
# EXPECTED_BYTES = 534904783
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/vgg-face.mat'
VGG_FILENAME = 'vgg-face.mat'
EXPECTED_BYTES = 1086058494


class FaceVgg(cnn_model.CnnModel):    
    def construct_network(self):
        print("begin to construct face-vgg......")
        
        with tf.name_scope('Input'):
            self.input_img = tf.placeholder(dtype=tf.float32, name='input_img',
                                            shape=[self.batch_size,self.width, self.height, 3])
            
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
        #self.conv2d_relu(self.fc6, 33, 'fc7')

        
    def get_data(self, video_paths, label_paths, gt_paths,window_size,clips,mode):
        print("create generator....")
        batch_gen = process_data.get_seq_batch(video_paths, label_paths, gt_paths, int(self.batch_size/window_size),window_size,clips=clips,mode=mode,width=self.width, height=self.height)    
        return batch_gen
       
    def inference(self):
        self.output = self.fc6
        
        
    def loss(self):
        with tf.name_scope('labels'):
            self.labels = tf.placeholder(dtype=tf.float32, name='ppg', shape=[self.batch_size, ])
        print("crteate loss-Function.....")
        with tf.name_scope('loss'):
            self.logits = tf.layers.dense(self.output, 1, None)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_sum(loss)
            
    def optimizer(self,lr,gstep):
        print("crteate optimizer.....")
        optimizer = tf.train.AdadeltaOptimizer(lr)
        self.opt = optimizer.minimize(self.loss, global_step=gstep)
    
    
    def evaluation(self):
        print("create evaluation methods.....")
        with tf.name_scope('hr_accuracy'):
            self.hrs = tf.placeholder(dtype=tf.float32, name='pred_hr', shape=[self.batch_size, ])
            self.gts = tf.placeholder(dtype=tf.float32, name='ground_truth', shape=[self.batch_size, ])
            diff = tf.abs(self.hrs - self.gts)
            indicator = tf.where(diff < 3)
            total_match = tf.cast(tf.size(indicator), tf.float32)
            self.hr_accuracy = tf.truediv(total_match, tf.cast(self.batch_size, tf.float32))   
        with tf.name_scope('sign_accuracy'):        
            label_signs = tf.cast(tf.greater(self.labels,0),tf.int32)
            right_signs = tf.equal( label_signs, self.pred_class )
            self.sign_accuracy = tf.reduce_mean(tf.cast(right_signs, tf.float32))  
           

    def create_summary(self):
        print("crteate summary.....")
        summary_class_loss = tf.summary.scalar('class_loss', self.loss) 
        summary_hr_accuracy = tf.summary.scalar('hr_accuracy', self.hr_accuracy)
        summary_train = tf.summary.merge([summary_class_loss, summary_sign_accuracy])
        summary_test = tf.summary.merge([summary_class_loss, summary_hr_accuracy]) 
        return summary_train, summary_test
        
        
if __name__ == '__main__':
    m = FaceVgg(64)
    m.construct_network(112,112)