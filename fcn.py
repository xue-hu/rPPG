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


class FCN(cnn_model.CnnModel):    
    def construct_network(self):
        print("begin to construct FCN......")
        
        with tf.name_scope('Input'):
            self.input_img = tf.placeholder(dtype=tf.float32, name='input_img',
                                            shape=[self.batch_size,self.width, self.height, 3])
            
        self.conv2d_relu(self.input_img, 0, 'conv1_1')
        self.conv2d_relu(self.conv1_1, 2, 'conv1_2')        
        self.avgpool(self.conv1_2, 'pool1', stream_name='s')
        self.dropout_layer(self.s_pool1)

        self.conv2d_relu(self.s_pool1, 4, 'conv2_1')
        self.conv2d_relu(self.conv2_1, 6, 'conv2_2')
        self.avgpool(self.conv2_2, 'pool2', stream_name='s')
        self.dropout_layer(self.s_pool2)

        self.conv2d_relu(self.s_pool2, 8, 'conv3_1')
        self.conv2d_relu(self.conv3_1, 10, 'conv3_2')
        self.conv2d_relu(self.conv3_2, 12, 'conv3_3')
        self.avgpool(self.conv3_3, 'pool3', stream_name='s')
        self.dropout_layer(self.s_pool3)
        
        self.conv2d_relu(self.s_pool3, 14, 'conv4_1')
        self.conv2d_relu(self.conv4_1, 16, 'conv4_2')
        self.conv2d_relu(self.conv4_2, 18, 'conv4_3')
        self.avgpool(self.conv4_3, 'pool4', stream_name='s')
        self.dropout_layer(self.s_pool4)
        
#         self.conv2d_relu(self.s_pool4, 20, 'conv5_1')
#         self.conv2d_relu(self.conv5_1, 22, 'conv5_2')
#         self.conv2d_relu(self.conv5_2, 24  , 'conv5_3')
        #self.avgpool(self.conv5_3, 'pool5', stream_name='s')
        #self.dropout_layer(self.s_pool5)
        
        #self.conv2d_relu(self.conv4_3, 26, 'fc6',trainable=True)
        #self.conv2d_relu(self.fc6, 28, 'fc7',trainable=True)

        
    def get_data(self, video_paths, label_paths, gt_paths,window_size,clips,mode):
        print("create generator....")
        batch_gen = process_data.get_seq_batch(video_paths, label_paths, gt_paths, int(self.batch_size/window_size),window_size,clips=clips,mode=mode,width=self.width, height=self.height)    
        return batch_gen
       
    def inference(self):
        batch, height, width, depth = self.s_pool4.shape.as_list()
        
        with tf.variable_scope('1x1_conv', reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable("weight", dtype=tf.float32, shape=[1, 1, depth, 1],
   initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable("bias", dtype=tf.float32, shape=[1, ],
                                initializer=tf.constant_initializer(0.0))
            z = tf.nn.conv2d(self.s_pool4, w, strides=[1, 1, 1, 1], padding='VALID') + b
            self.output = tf.layers.flatten( tf.nn.tanh(z) )  

#         with tf.variable_scope('GAP', reuse=tf.AUTO_REUSE) as scope:            
#             z = tf.layers.average_pooling2d(self.s_pool4,pool_size=[height, width],strides=[1, 1], padding='VALID')
#             self.output = tf.layers.flatten( tf.nn.tanh(z) )  

        #self.output = self.fc6
        
        
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
    m = FCN(64,112, 112, model='fcn')
    m.construct_network()
        