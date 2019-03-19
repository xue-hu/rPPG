#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import tensorflow as tf
import utils
import scipy.io
import numpy as np
import math
import cv2

# VGG-19 parameters file
N_CLASSES = 2
FRAME_RATE = 30.0
# VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
# VGG_FILENAME = 'imagenet-vgg-verydeep-19.mat'
# EXPECTED_BYTES = 534904783
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/vgg-face.mat'
VGG_FILENAME = 'vgg-face.mat'
EXPECTED_BYTES = 1086058494


class CnnModel(object):
    def __init__(self, batch_size,img_width, img_height):
        self.batch_size = batch_size
        self.width = img_width
        self.height = img_height
        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob', shape=[])            
        utils.download(VGG_DOWNLOAD_LINK, VGG_FILENAME, EXPECTED_BYTES)
        self.vgg = scipy.io.loadmat(VGG_FILENAME)['layers']
        self.t = 0

    def __vgg_weights(self, lyr_idx, lyr_name):
        w = self.vgg[0][lyr_idx][0][0][2][0][0]
        b = self.vgg[0][lyr_idx][0][0][2][0][1]        
        if VGG_FILENAME == 'vgg-face.mat':
            name = self.vgg[0][lyr_idx][0][0][1][0]
        else:
            name = self.vgg[0][lyr_idx][0][0][0][0]
        assert lyr_name == name
        return w, b.reshape(b.size)

    def conv2d_tanh(self, pre_lyr, out_dims, lyr_name, kernel_size=3, lyr_idx=1, stream_name='d'):
        if lyr_idx == 0:
            batch_size, height, width, depth = pre_lyr.shape
        else:
            batch_size, height, width, depth = pre_lyr.shape.as_list()

        with tf.variable_scope((stream_name+'_'+lyr_name), reuse=tf.AUTO_REUSE) as scope:            
            w = tf.get_variable("weight", dtype=tf.float32, shape=[kernel_size, kernel_size, depth, out_dims],
   initializer=tf.random_normal_initializer(stddev=0.4))
            b = tf.get_variable("bias", dtype=tf.float32, shape=[out_dims,], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(pre_lyr, w, strides=[1, 1, 1, 1], padding='SAME')
            out = tf.nn.tanh((conv + b), name=scope.name)
            
        print(lyr_name)
        print(w.shape)
        print(out.shape)
        a = w.shape.as_list()
        d = a[0] * a[1] * a[2] * a[3]
        self.t += d
        print(d)
        print(self.t)
        setattr(self, (stream_name + '_' + lyr_name), out)
        
    def conv2d_relu(self, pre_lyr, lyr_idx, lyr_name, kernel_size=3):        
        w_init, b_init = self.__vgg_weights(lyr_idx, lyr_name)
        w_init = tf.convert_to_tensor(w_init, dtype=tf.float32)
        b_init = tf.convert_to_tensor(b_init, dtype=tf.float32)
        if lyr_idx > 30:
            pad = 'VALID'
        else:
            pad = 'SAME'
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable(name="weight", dtype=tf.float32, initializer=w_init,trainable=False)
            b = tf.get_variable(name="bias", dtype=tf.float32, initializer=b_init,trainable=False)                
            conv = tf.nn.conv2d(pre_lyr, w, strides=[1, 1, 1, 1], padding=pad)
            out = tf.nn.relu((conv + b), name=scope.name)
            
        print(lyr_name)
        print(w.shape)
        print(out.shape)
        a = w.shape.as_list()
        d = a[0] * a[1] * a[2] * a[3]
        self.t += d
        print(d)
        print(self.t)
        setattr(self, (lyr_name), out)
    

    def fully_connected_layer(self, pre_lyr, out_dims, lyr_name,last_lyr=False):
        _, height, width, depth = pre_lyr.shape.as_list()
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable("weight", dtype=tf.float32, shape=[height, width, depth, out_dims],
   initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable("bias", dtype=tf.float32, shape=[out_dims, ],
                                initializer=tf.constant_initializer(0.0))
            z = tf.nn.conv2d(pre_lyr, w, strides=[1, 1, 1, 1], padding='VALID') + b
            if not last_lyr:
                out = tf.nn.tanh(z, name=scope.name)
            else:
                out = z
        print(lyr_name)
        print(w.shape)
        print(out.shape)
        a = w.shape.as_list()
        d = a[0]*a[1]*a[2]*a[3]
        self.t += d
        print(d)
        print(self.t)
        setattr(self, lyr_name, out)
        
        
    def attention_layer(self, dy_lyr, sta_lyr, dy_lyr_name, sta_lyr_name):
        batch_size, width, height, depth = sta_lyr.shape.as_list()
        with tf.variable_scope('atten_'+sta_lyr_name[2:], reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable("weight", dtype=tf.float32, initializer=tf.random_normal([1, 1, depth, 1], stddev=0.4))
            b = tf.get_variable("bias", dtype=tf.float32, initializer=tf.zeros([1, ]))
            conv = tf.nn.conv2d(sta_lyr, w, strides=[1, 1, 1, 1], padding='SAME')
            mask = tf.nn.sigmoid(conv + b, name='mask')
            l1_norm = tf.reduce_sum(tf.abs(mask)) 
            mask = batch_size*width * height * mask / (2.0 * l1_norm)            
            #mask = self.softmax(conv + b, axis=[1,2]) 
            out = tf.multiply(dy_lyr,mask, name="masked_feat")
            
            print('atten_'+sta_lyr_name[2:])
            print(w.shape)
            print(b.shape)
            print(conv.shape)
            print(out.shape)      
            self.t += depth
            print(depth)
            print(self.t)
            
            setattr(self, ('atten_' + sta_lyr_name[2:]+'_mask'), mask)
            setattr(self, ('atten_' + sta_lyr_name[2:]), out)

        

    def maxpool(self, prev_lyr, lyr_name):
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            out = tf.nn.max_pool(prev_lyr, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pol')
        setattr(self,lyr_name, out)

    def avgpool(self, prev_lyr, lyr_name, stream_name='d'):
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            out = tf.nn.avg_pool(prev_lyr, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pol')
        print(lyr_name)
        print(prev_lyr.shape)
        print(out.shape)
        setattr(self, (stream_name + '_' + lyr_name), out)

    
    def dropout_layer(self, dy_lyr):
        dy_lyr = tf.nn.dropout(dy_lyr, self.keep_prob)


        

if __name__ == '__main__':
    frame = cv2.imread('./processed_video/101/Proband01/1/0.jpg').astype(np.float32)
    diff = cv2.imread('./processed_video/101/Proband01/1/1.jpg').astype(np.float32)
    frame = np.expand_dims(frame, 0)
    diff = np.expand_dims(diff, 0)
    model = CnnModel(frame, diff, 1)
    model.vgg_face_load()
    #print(model.t)
