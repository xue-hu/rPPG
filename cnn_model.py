#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import tensorflow as tf
import utils
import scipy.io
import numpy as np
import math
import cv2
from stn import spatial_transformer_network as transformer

# VGG-19 parameters file
N_CLASSES = 2
FRAME_RATE = 30.0

VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_FILENAME = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783

FACE_VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/vgg-face.mat'
FACE_VGG_FILENAME = 'vgg-face.mat'
FACE_VGG_EXPECTED_BYTES = 1086058494

FCN_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/pascal-fcn8s-tvg-dag.mat'
FCN_FILENAME = 'pascal-fcn8s-tvg-dag.mat'
FCN_EXPECTED_BYTES = 500082003
# REGULARIZER=tf.contrib.layers.l2_regularizer(0.001)

class CnnModel(object):
    def __init__(self, lr,batch_size,length, img_width, img_height, model='face_vgg'):
        self.batch_size = batch_size
        self.width = img_width
        self.height = img_height
        self.model = model
        self.length = length
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.lr = lr
        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob', shape=[]) 
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training') 
        if model == 'face_vgg':
            download_link = FACE_VGG_DOWNLOAD_LINK
            filename = FACE_VGG_FILENAME
            expected_bytes = FACE_VGG_EXPECTED_BYTES
        elif model == 'deep_vgg':
            download_link = VGG_DOWNLOAD_LINK
            filename = VGG_FILENAME
            expected_bytes = VGG_EXPECTED_BYTES
        else:
            download_link = FCN_DOWNLOAD_LINK
            filename = FCN_FILENAME
            expected_bytes = FCN_EXPECTED_BYTES
            
        utils.download(download_link, filename, expected_bytes)
        self.vgg = scipy.io.loadmat(filename)
        self.t = 0

    def __vgg_weights(self, lyr_idx, lyr_name):
             
        w = self.vgg['layers'][0][lyr_idx][0][0][2][0][0]
        b = self.vgg['layers'][0][lyr_idx][0][0][2][0][1]
        o_dim = min(int(w.shape[-1] / 2), 64)
        i_dim = min(int(w.shape[-2]) if (lyr_idx == 0) else int(w.shape[-2] / 2),64)
        w = w[:, :, :i_dim, :o_dim]
        b = b[:o_dim]
        if self.model=='face_vgg':
            name = self.vgg['layers'][0][lyr_idx][0][0][1][0]
        else:
            name = self.vgg['layers'][0][lyr_idx][0][0][0][0]
        #assert lyr_name in name
        return w, b.reshape(b.size)
    

    def conv2d_layer(self, pre_lyr, lyr_idx, out_dims, lyr_name, kernel_size_x=3, kernel_size_y=3,stride_x=1,stride_y=1,padding='SAME', stream_name='d',activation=tf.nn.relu,batch_norm=False, trained=True,linear=False): 
       
        with tf.variable_scope((stream_name+'_'+lyr_name), reuse=tf.AUTO_REUSE,regularizer=tf.contrib.layers.l2_regularizer(0.001)) as scope:
            if trained:
                w_init, b_init = self.__vgg_weights(lyr_idx, lyr_name)
                w_init = tf.convert_to_tensor(w_init, dtype=tf.float32)
                b_init = tf.convert_to_tensor(b_init, dtype=tf.float32)
                w = tf.get_variable(name="weight", dtype=tf.float32, initializer=w_init)
                b = tf.get_variable(name="bias", dtype=tf.float32, initializer=b_init) 
            else:
                if lyr_idx == 0:
                    batch_size, height, width, depth = pre_lyr.shape
                else:
                    batch_size, height, width, depth = pre_lyr.shape.as_list()
                w = tf.get_variable("weight", dtype=tf.float32, shape=[kernel_size_x, kernel_size_y, depth, out_dims],
       initializer=tf.random_normal_initializer(stddev=0.1))
                b = tf.get_variable("bias", dtype=tf.float32, shape=[out_dims,], initializer=tf.constant_initializer(0.0))
            
            conv = tf.nn.conv2d(pre_lyr, w, strides=[1, stride_x, stride_y, 1], padding=padding, name=scope.name)
            
            if batch_norm:
                conv = tf.layers.batch_normalization(conv, training = self.is_training)
            if not linear:
                out = activation((conv + b), name=scope.name)
            else:
                out = conv + b
            
        print(lyr_name)
        print(w.shape)
        print(out.shape)
        a = w.shape.as_list()
        d = a[0] * a[1] * a[2] * a[3]
        self.t += d
        print(d)
        print(self.t)
        if hasattr(self,  (stream_name+'_'+lyr_name)):
            setattr(self, (stream_name+'_'+lyr_name+'_1'), out)
        else:
            setattr(self, (stream_name+'_'+lyr_name), out)
            
            
    def trans_conv2d_layer(self, pre_lyr, lyr_idx, out_dims, lyr_name, kernel_size_x=3, kernel_size_y=3,stride_x=2,stride_y=2,padding='SAME', stream_name='d',activation=tf.nn.relu,batch_norm=False, trained=True,linear=False): 
       
        with tf.variable_scope((stream_name+'_'+lyr_name), reuse=tf.AUTO_REUSE,regularizer=tf.contrib.layers.l2_regularizer(0.001)) as scope:
            if trained:
                w_init, b_init = self.__vgg_weights(lyr_idx, lyr_name)
                w_init = tf.convert_to_tensor(w_init, dtype=tf.float32)
                b_init = tf.convert_to_tensor(b_init, dtype=tf.float32)
                w = tf.get_variable(name="weight", dtype=tf.float32, initializer=w_init)
                b = tf.get_variable(name="bias", dtype=tf.float32, initializer=b_init) 
            else:
                if lyr_idx == 0:
                    batch_size, height, width, depth = pre_lyr.shape
                else:
                    batch_size, height, width, depth = pre_lyr.shape.as_list()
                w = tf.get_variable("weight", dtype=tf.float32, shape=[kernel_size_x, kernel_size_y,out_dims,depth],
       initializer=tf.random_normal_initializer(stddev=0.1))
                b = tf.get_variable("bias", dtype=tf.float32, shape=[out_dims,], initializer=tf.constant_initializer(0.0))
            
            conv = tf.nn.conv2d_transpose(pre_lyr, w, output_shape=[int(batch_size), int(height*2), int(width*2), int(out_dims)],strides=[1,stride_x, stride_y, 1], padding=padding, name=scope.name)
            
            if batch_norm:
                conv = tf.layers.batch_normalization(conv, training = self.is_training)
            if not linear:
                out = activation((conv + b), name=scope.name)
            else:
                out = conv + b
            
        print(lyr_name)
        print(w.shape)
        print(out.shape)
        a = w.shape.as_list()
        d = a[0] * a[1] * a[2] * a[3]
        self.t += d
        print(d)
        print(self.t)
        if hasattr(self,  (stream_name+'_'+lyr_name)):
            setattr(self, (stream_name+'_'+lyr_name+'_1'), out)
        else:
            setattr(self, (stream_name+'_'+lyr_name), out)
        
    

    def fully_connected_layer(self, pre_lyr, out_dims, lyr_name,activation=tf.nn.tanh,batch_norm=False, last_lyr=False):
        _, height, width, depth = pre_lyr.shape.as_list()
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE,regularizer=tf.contrib.layers.l2_regularizer(0.001)) as scope:
            w = tf.get_variable("weight", dtype=tf.float32, shape=[height, width, depth, out_dims],
    initializer=tf.random_normal_initializer(stddev=0.4))
            b = tf.get_variable("bias", dtype=tf.float32, shape=[out_dims, ],
                                initializer=tf.constant_initializer(0.0))

            conv = tf.nn.conv2d(pre_lyr, w, strides=[1, 1, 1, 1], padding='VALID', name=scope.name) 

            if batch_norm:
                conv = tf.layers.batch_normalization(conv, training = self.is_training)
            if not last_lyr:
                out = activation(conv + b, name=scope.name)
            else:
                out = conv + b
         
        print(lyr_name)
        print(w.shape)
        print(out.shape)
        a = w.shape.as_list()
        d = a[0]*a[1]*a[2]*a[3]
        self.t += d
        print(d)
        print(self.t)
        if hasattr(self, lyr_name):
            setattr(self, lyr_name+'_1', out)
        else:
            setattr(self, lyr_name, out)
            
            
    
    def auto_encoder(self, pre_lyr,lyr_name,batch_norm=False, lyr_norm=False, linear=True):
        batch_size, height, width, depth = pre_lyr.shape.as_list()
        x = tf.reshape(pre_lyr, shape=[batch_size, height*width*depth])
        if self.keep_prob == 1:
            keep_prob = 1
        else:
            keep_prob = 0.8
            
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE,regularizer=tf.contrib.layers.l2_regularizer(0.001)) as scope:
            encode_1 = tf.layers.dense( x, 1024,activation=tf.nn.elu,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='encode_1')
            if batch_norm:
                encode_1 = tf.layers.batch_normalization(encode_1, training=self.is_training)
            encode_1 = tf.nn.dropout( encode_1, self.keep_prob)
            encode_1 = tf.nn.tanh(encode_1)            
            
            encode_2 = tf.layers.dense(encode_1,256,activation=tf.nn.elu,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='encode_2')
            if batch_norm:
                encode_2 = tf.layers.batch_normalization(encode_2, training=self.is_training)
            encode_2 = tf.nn.dropout( encode_2, self.keep_prob)
            encode_2 = tf.nn.tanh(encode_2) 
            
            encode = tf.layers.dense( encode_2,16,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='encode_0')
            
            decode_2 = tf.layers.dense(encode,256,activation=tf.nn.elu,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='decode_2')
            if batch_norm:
                decode_2 = tf.layers.batch_normalization(decode_2, training=self.is_training)
            decode_2 = tf.nn.dropout( decode_2, self.keep_prob)
            decode_2 = tf.nn.tanh(decode_2) 
            
            decode_1 = tf.layers.dense( decode_2, 1024, activation=tf.nn.elu,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='decode_1')
            if batch_norm:
                decode_1 = tf.layers.batch_normalization(encode_1, training=self.is_training)
            decode_1 = tf.nn.tanh(decode_1)   
            
            decode = tf.layers.dense( decode_1, height*width*depth, activation=tf.nn.tanh,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='decode_0')
            
            
#             w = tf.get_variable("weight", dtype=tf.float32, shape=[height,width,depth, out_dims],
#     initializer=tf.random_normal_initializer(stddev=0.4))
#             b = tf.get_variable("bias", dtype=tf.float32, shape=[out_dims, ],
#                                 initializer=tf.constant_initializer(0.0))
#             encode = tf.nn.conv2d(pre_lyr, w, strides=[1, 1, 1, 1], padding='VALID', name=scope.name) 
#             if batch_norm:
#                 encode = tf.layers.batch_normalization(encode, training = self.is_training)
#             if not linear:
#                 encode = tf.nn.tanh(encode + b)
#             else:
#                 encode = encode + b
#             encode = tf.nn.dropout(encode, keep_prob)
            
#         with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE,regularizer=tf.contrib.layers.l2_regularizer(0.001)) as scope:
#             b_inv = tf.Variable(tf.constant(0.0, shape=[depth, ], dtype=tf.float32))
#             decode = tf.nn.conv2d_transpose(encode, w, output_shape=[self.batch_size, height, width, depth],
#                                 strides=[1, 1, 1, 1], padding='VALID', name=scope.name) 
#             if not linear:
#                 decode = tf.nn.tanh(decode + b_inv)
#             else:
#                 decode = decode + b_inv
#             decode = tf.nn.dropout(decode, keep_prob)
             
         
        print(lyr_name)
#         print(w.shape)
        print(encode.shape)
        print(decode.shape)
#         a = w.shape.as_list()
        d = 512 * height * width * depth
        self.t += d
        print(d)
        print(self.t)
        setattr(self, 'encode', encode)
        setattr(self, 'decode', decode)
        
        
    def attention_layer(self, dy_lyr, sta_lyr, lyr_name):
        batch_size, width, height, depth = sta_lyr.shape.as_list()
        with tf.variable_scope('atten_'+lyr_name, reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable("weight", dtype=tf.float32, initializer=tf.random_normal([1, 1, depth, 1], stddev=0.4))
            b = tf.get_variable("bias", dtype=tf.float32, initializer=tf.zeros([1, ]))
            conv = tf.nn.conv2d(sta_lyr, w, strides=[1, 1, 1, 1], padding='SAME', name=scope.name)
            mask = tf.nn.sigmoid(conv + b, name='mask')
            l1_norm = tf.reduce_sum(tf.abs(mask)) 
            mask = batch_size*width * height * mask / (2.0 * l1_norm)            
            #mask = self.softmax(conv + b, axis=[1,2]) 
            out = tf.multiply(dy_lyr,mask, name="masked_feat")
            
            print('atten_'+lyr_name)
            print(w.shape)
            print(out.shape)      
            self.t += depth
            print(depth)
            print(self.t)
            
            setattr(self, ('atten_' + lyr_name+'_mask'), mask)
            setattr(self, ('atten_' + lyr_name), out)

        

    def maxpool(self, prev_lyr, lyr_name):
        _, width, height, _ = prev_lyr.shape.as_list()
        kernel_size_x = min(2,width)
        kernel_size_y = min(2,height)
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            out = tf.nn.max_pool(prev_lyr, [1, kernel_size_x, kernel_size_y, 1], [1, 2, 2, 1], padding='VALID', name=(stream_name + '_' + lyr_name))
        setattr(self,lyr_name, out)

    def avgpool(self, prev_lyr, lyr_name, stream_name='d'):
        _, width, height, _ = prev_lyr.shape.as_list()
        kernel_size_x = min(2,width)
        kernel_size_y = min(2,height)
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            out = tf.nn.avg_pool(prev_lyr, [1, kernel_size_x, kernel_size_y, 1], [1, 2, 2, 1], padding='VALID', name=(stream_name + '_' + lyr_name))
            
        print(lyr_name)
        print(prev_lyr.shape)
        print(out.shape)
        if hasattr(self,  (stream_name+'_'+lyr_name)):
            setattr(self, (stream_name+'_'+lyr_name+'_1'), out)
        else:
            setattr(self, (stream_name+'_'+lyr_name), out)

    
    def globalpool(self, prev_lyr,activation='max'):
        _, width, height, _ = prev_lyr.shape.as_list()
        with tf.variable_scope('GlobalPool', reuse=tf.AUTO_REUSE) as scope:
            if activation == 'max':
                out = tf.layers.max_pooling2d(prev_lyr,(height,width),(1,1),name=scope.name)
            elif activation == 'avg':
                out = tf.layers.average_pooling2d(prev_lyr,(height,width),(1,1),name=scope.name)
        mean = tf.reduce_mean(out,axis=[0,1,2],keep_dims=True)
        out = tf.divide(out, mean)
        mean, var = tf.nn.moments(out, [0,1,2], keep_dims=True)
        out = tf.div(tf.subtract(out, mean), tf.sqrt(var))
        return out
        
    
    def dropout_layer(self, dy_lyr):
        dy_lyr = tf.nn.dropout(dy_lyr, self.keep_prob)
        
    
    def bottleneck_layer(self, pre_lyr, lyr_name, outdims=1,linear=True, lyr_norm=False,batch_norm=False):
        with tf.variable_scope('bottleneck', reuse=tf.AUTO_REUSE) as scope:
            _, height, width, depth = pre_lyr.shape.as_list()
            pre_lyr = tf.reshape(pre_lyr,(self.batch_size,self.length, width, height,depth))
            feat_map = tf.reshape(pre_lyr,(self.batch_size,self.length, width*height*depth,1))
            out = feat_map
            if lyr_norm:
                out = tf.contrib.layers.layer_norm(out)
            if batch_norm:
                out = tf.layers.batch_normalization(out, training = self.is_training)
        
        print('bottleneck_'+lyr_name)
        print(out.shape) 
        setattr(self, ('bottleneck_'+lyr_name), out)
        
    def conv3d_layer(self, pre_lyr, lyr_name, outdims=1,linear=True):
        batch_size, length, height, width, depth = pre_lyr.shape.as_list()
        feat_map = tf.reshape(pre_lyr, shape=[batch_size, height, width, depth*length])
        value = [-1]* int(depth*length/2)  + [1]* int(depth*length/2)
        init = tf.constant_initializer(value)
        with tf.variable_scope('bottleneck', reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable("weight", dtype=tf.float32, shape=[1, 1, depth*length, outdims],
   initializer=init)
            b = tf.get_variable("bias", dtype=tf.float32, shape=[outdims, ],
                                initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(feat_map, w, strides=[1, 1, 1, 1], padding='SAME', name=scope.name) + b
            if linear:
                out = conv
            else:
                out = tf.nn.tanh(conv, name=scope.name)       
        print('3d_'+lyr_name)
        print(feat_map.shape)
        print(w.shape)
        print(out.shape) 
        setattr(self, ('conv3d_'+lyr_name), out)
    
    
    def st_layer(self, pre_lyr,lyr_name, lyr_idx = 0):
        n_fc = 9
        if lyr_idx == 0:
            batch_size, height, width, depth = pre_lyr.shape
        else:
            batch_size, height, width, depth = pre_lyr.shape.as_list()

        initial = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        initial = initial.astype('float32').flatten()
        b_init = tf.constant_initializer(initial)

        with tf.variable_scope(('stn_'+lyr_name), reuse=tf.AUTO_REUSE) as scope:
        # localization network
            x = tf.reshape(pre_lyr, [batch_size, -1])
            W_fc1 = tf.get_variable(name='W_fc1',dtype=tf.float32,shape=[height*width*depth, n_fc], initializer=tf.zeros_initializer())
            b_fc1 = tf.get_variable(name='b_fc1',dtype=tf.float32, shape=[n_fc,], initializer=b_init)
            h_fc1 = tf.nn.tanh(tf.matmul(x, W_fc1) + b_fc1, name=scope.name)
            h_fc1 = tf.reshape(h_fc1, shape=[batch_size, 1, 1, 3, 3])
            h_fc1 = tf.tile(h_fc1,[1,height, width,1,1])
            x_out = tf.reshape(pre_lyr, [batch_size, height, width, 1, depth])
            # spatial transformer layer
            h_trans = tf.reshape(tf.matmul( x_out,h_fc1),[batch_size, height, width,depth])
            
        print(lyr_name)
        print(W_fc1.shape)
        print(h_trans.shape)
        a = W_fc1.shape.as_list()
        d = a[0]*a[1]
        self.t += d
        print(d)
        print(self.t)
        
        if hasattr(self, (lyr_name)):
            setattr(self, (lyr_name+'_1'), h_trans)
        else:
            setattr(self, (lyr_name), h_trans)

        

if __name__ == '__main__':
    frame = cv2.imread('./processed_video/101/Proband01/1/0.jpg').astype(np.float32)
    diff = cv2.imread('./processed_video/101/Proband01/1/1.jpg').astype(np.float32)
    frame = np.expand_dims(frame, 0)
    diff = np.expand_dims(diff, 0)
    model = CnnModel(frame, diff, 1)
    model.vgg_face_load()
    #print(model.t)
