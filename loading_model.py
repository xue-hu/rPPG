#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import tensorflow as tf
import utils
import scipy.io
import numpy as np
import cv2

# VGG-19 parameters file
N_CLASSES = 2
# VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
# VGG_FILENAME = 'imagenet-vgg-verydeep-19.mat'
# EXPECTED_BYTES = 534904783
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/vgg-face.mat'
VGG_FILENAME = 'vgg-face.mat'
EXPECTED_BYTES = 1086058494


class NnModel(object):
    def __init__(self, img, diff, keep_prob):
        self.input_img = img
        self.input_diff = diff
        self.keep_prob = keep_prob
        utils.download(VGG_DOWNLOAD_LINK, VGG_FILENAME, EXPECTED_BYTES)
        self.vgg = scipy.io.loadmat(VGG_FILENAME)['layers']
        self.t = 0

    def __vgg_weights(self, lyr_idx, lyr_name):
        w = self.vgg[0][lyr_idx][0][0][2][0][0]
        b = self.vgg[0][lyr_idx][0][0][2][0][1]
        o_dim = int(w.shape[-1] / 2)
        i_dim = int(w.shape[-2]) if (lyr_idx == 0) else int(w.shape[-2] / 2)
        w = w[:, :, :i_dim, :o_dim]
        b = b[:o_dim]
        if VGG_FILENAME == 'vgg-face.mat':
            name = self.vgg[0][lyr_idx][0][0][1][0]
        else:
            name = self.vgg[0][lyr_idx][0][0][0][0]
        assert lyr_name == name
        return w, b.reshape(b.size)
    
    def softmax(self, target, axis):
        max_axis = tf.reduce_max(target, axis, keepdims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
        softmax = target_exp / normalize
        return softmax

    def conv2d_tanh(self, pre_lyr, lyr_idx, out_dims, lyr_name, stream_name='d'):
        w_init, b_init = self.__vgg_weights(lyr_idx, lyr_name)
        w_init = tf.convert_to_tensor(w_init, dtype=tf.float32)
        b_init = tf.convert_to_tensor(b_init, dtype=tf.float32)
        if lyr_idx == 0:
            batch_size, height, width, depth = pre_lyr.shape
        else:
            batch_size, height, width, depth = pre_lyr.shape.as_list()
        with tf.variable_scope((stream_name+'_'+lyr_name), reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable(name="weight", dtype=tf.float32, initializer=w_init)
            b = tf.get_variable(name="bias", dtype=tf.float32, initializer=b_init)     
#             w = tf.get_variable("weight", dtype=tf.float32, shape=[3, 3, depth, out_dims],
#    initializer=tf.random_normal_initializer(stddev=0.4))
#             b = tf.get_variable("bias", dtype=tf.float32, shape=[out_dims,], initializer=tf.constant_initializer(0.0))
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

    def fully_connected_layer(self, pre_lyr, out_dims, lyr_name, last_lyr=False):
        _, height, width, depth = pre_lyr.shape.as_list()
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable("weight", dtype=tf.float32, shape=[height, width, depth, out_dims],
   initializer=tf.random_normal_initializer(stddev=0.1))
            b = tf.get_variable("bias", dtype=tf.float32, shape=[out_dims, ],
                                initializer=tf.constant_initializer(0.0))
            z = tf.nn.conv2d(pre_lyr, w, strides=[1, 1, 1, 1], padding='VALID') + b
            if not last_lyr:
                out = tf.nn.tanh(z, name=scope.name)
               #out = tf.nn.relu(z, name=scope.name)
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

        

    def maxpool(self, prev_lyr, lyr_name, stream_name='d'):
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            out = tf.nn.max_pool(prev_lyr, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pol')
        setattr(self, (stream_name + '_' + lyr_name), out)

    def avgpool(self, prev_lyr, lyr_name, stream_name='d'):
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            out = tf.nn.avg_pool(prev_lyr, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pol')
        print(lyr_name)
        print(prev_lyr.shape)
        print(out.shape)
        setattr(self, (stream_name + '_' + lyr_name), out)

    
    def dropout_layer(self, dy_lyr):
        dy_lyr = tf.nn.dropout(dy_lyr, self.keep_prob)

    def two_stream_vgg_load(self):
        print("begin to construct two stream vgg......")

        self.conv2d_tanh(self.input_diff, 0, 32, 'conv1_1')
        self.conv2d_tanh(self.d_conv1_1, 2, 32, 'conv1_2')
        
        self.conv2d_tanh(self.input_img, 0, 32, 'conv1_1', stream_name='s')
        self.conv2d_tanh(self.s_conv1_1, 2, 32, 'conv1_2', stream_name='s')
        
        self.attention_layer(self.d_conv1_2, self.s_conv1_2, 'd_conv1_2', 's_conv1_2')
        self.avgpool(self.atten_conv1_2, 'pool1')
        self.avgpool(self.s_conv1_2, 'pool1', stream_name='s')
        self.dropout_layer(self.d_pool1)

        

        self.conv2d_tanh(self.d_pool1, 5, 64, 'conv2_1')
        self.conv2d_tanh(self.d_conv2_1, 7, 64, 'conv2_2')
        
        self.conv2d_tanh(self.s_pool1, 5, 64, 'conv2_1', stream_name='s')
        self.conv2d_tanh(self.s_conv2_1, 7, 64, 'conv2_2', stream_name='s')
        
        self.attention_layer(self.d_conv2_2, self.s_conv2_2, 'd_conv2_2', 's_conv2_2')
        self.avgpool(self.atten_conv2_2, 'pool2')
        self.dropout_layer(self.d_pool2)

      #  self.conv2d_tanh(self.d_pool2, 10, 64, 'conv3_1')
        #self.avgpool(self.d_conv3_1, 'conv3_1')
       # self.conv2d_tanh(self.d_conv3_1, 12, 64, 'conv3_2')
       # self.conv2d_tanh(self.d_conv3_2, 14, 'conv3_3')
       # self.conv2d_tanh(self.d_conv3_3, 16, 'conv3_4')
       # self.conv2d_tanh(self.s_pool2, 10, 64, 'conv3_1', stream_name='s')
       # self.conv2d_tanh(self.s_conv3_1, 12, 64, 'conv3_2', stream_name='s')
       # self.conv2d_tanh(self.s_conv3_2, 14, 'conv3_3', stream_name='s')
       # self.conv2d_tanh(self.s_conv3_3, 16, 'conv3_4', stream_name='s')
       # self.attention_layer(self.d_conv3_1, self.s_conv3_1, 'd_conv3_1', 's_conv3_1')
       # self.avgpool(self.atten_conv3_1, 'pool3')
     #   self.avgpool(self.d_conv3_1, 'pool3')
       # self.avgpool(self.s_conv3_2, 'pool3', stream_name='s')
       # self.dropout_layer(self.d_pool3)

        self.fully_connected_layer(self.d_pool2, 128, 'fc7')
        self.dropout_layer(self.fc7)
        #self.fully_connected_layer(self.fc6, 128, 'fc7')
        self.fully_connected_layer(self.fc7, 1, 'reg_output', last_lyr=True)
        self.fully_connected_layer(self.fc7, N_CLASSES, 'class_output', last_lyr=True)
        print('param nums: '+ str(self.t))
        print("done.")

if __name__ == '__main__':
    frame = cv2.imread('D:\PycharmsProject\yutube8M/0.jpg').astype(np.float32)
    diff = cv2.imread('D:\PycharmsProject\yutube8M/1.jpg').astype(np.float32)
    frame = np.expand_dims(frame, 0)
    diff = np.expand_dims(diff, 0)
    model = NnModel(frame, diff, 1)
    model.two_stream_vgg_load()
    #print(model.t)
