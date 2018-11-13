__author__ = 'Iris'

import os
import tensorflow as tf
import utils
import scipy.io
import numpy as np
import cv2

# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_FILENAME = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783


class NnModel(object):
    def __init__(self, img, diff):
        self.input_img = img
        self.input_diff = diff
        utils.download(VGG_DOWNLOAD_LINK, VGG_FILENAME, EXPECTED_BYTES)
        self.vgg = scipy.io.loadmat(VGG_FILENAME)['layers']
        self.mean_pixels = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))

    def __vgg_weights(self, lyr_idx, lyr_name):
        w = self.vgg[0][lyr_idx][0][0][2][0][0]
        b = self.vgg[0][lyr_idx][0][0][2][0][1]
        assert lyr_name == self.vgg[0][lyr_idx][0][0][0][0]
        return w, b.reshape(b.size)  #####why not b???? ValueError: Dimensions must be equal, but are 492 and 64 for 'add' (op: 'Add') with input shapes: [1,492,492,64], [64,1].

    def conv2d_relu(self, pre_lyr, lyr_idx, lyr_name, stream_name='d'):
        w_init, b_init = self.__vgg_weights(lyr_idx,lyr_name)
        w_init = tf.convert_to_tensor(w_init, dtype=tf.float32)
        b_init = tf.convert_to_tensor(b_init, dtype=tf.float32)
        #print(w_init.shape)
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable("weight", dtype=tf.float32, initializer=w_init)
            b = tf.get_variable("bias", dtype=tf.float32, initializer=b_init)
        conv = tf.nn.conv2d(pre_lyr, w, strides=[1, 1, 1, 1], padding='SAME')
        out = tf.nn.relu((conv + b), name=scope.name)
        setattr(self, (stream_name+'_'+lyr_name), out)

    def conv2d_tahn(self, pre_lyr, lyr_idx, lyr_name, stream_name='d'):
        w_init, b_init = self.__vgg_weights(lyr_idx,lyr_name)
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable("weight", dtype=tf.float32, initializer=w_init)
            b = tf.get_variable("bias", dtype=tf.float32, initializer=b_init)
        conv = tf.nn.conv2d(pre_lyr, w, strides=[1, 1, 1, 1], padding='SAME')
        out = tf.nn.tanh((conv + b), name=scope.name)
        setattr(self, (stream_name+'_'+lyr_name), out)

    # def regression_output_layer(self,pre_lyr, lyr_name):
    #     with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
    #         w = tf.get_variable("weight", dtype=tf.float32, initializer= tf.random_normal( pre_lyr.shape) )
    #         b = tf.get_variable("bias", dtype=tf.float32, initializer=tf.zeros([1,]) )
    #     z = tf.reduce_sum( tf.multiply(pre_lyr, w) ) + b
    #     out = tf.nn.relu( z , name=scope.name)
    #     setattr(self, lyr_name, out)

    def maxpool(self, prev_lyr, lyr_name, stream_name='d'):
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            out = tf.nn.max_pool(prev_lyr, [1, 2, 2, 1] , [1, 2, 2, 1], padding='VALID', name='pol')
        setattr(self, (stream_name+'_'+lyr_name), out)

    def avgpool(self, prev_lyr, lyr_name, stream_name='d'):
        with tf.variable_scope(lyr_name, reuse=tf.AUTO_REUSE) as scope:
            out = tf.nn.avg_pool(prev_lyr, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pol')
        setattr(self, (stream_name+'_'+lyr_name), out)

    def attention_layer(self, dy_lyr, sta_lyr, dy_lyr_name, sta_lyr_name):
        _, width, height, depth = sta_lyr.shape.as_list()
        #print(width, height, depth)
        with tf.variable_scope(sta_lyr_name, reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable("weight", dtype=tf.float32, initializer=tf.random_normal([1, 1, depth, 1]))
            b = tf.get_variable("bias", dtype=tf.float32, initializer=tf.zeros([1, ]))
        conv = tf.nn.conv2d(sta_lyr, w, strides=[1, 1, 1, 1], padding='SAME')
        l1_norm = tf.norm(conv, ord=1)
        mask = width*height*tf.nn.sigmoid(conv+b, name='mask') / (2.0 * l1_norm)
        out = tf.multiply(mask, dy_lyr, name="masked_feat")
        dy_lyr = out



    def two_stream_vgg_load(self):
        print("begin to construct two stream vgg......")

        self.conv2d_relu(self.input_diff, 0, 'conv1_1')
        self.conv2d_relu(self.d_conv1_1, 2, 'conv1_2')
        self.conv2d_relu(self.input_img, 0, 'conv1_1', stream_name='s')
        self.conv2d_relu(self.s_conv1_1, 2, 'conv1_2', stream_name='s')
        self.attention_layer(self.d_conv1_2, self.s_conv1_2, 'd_conv1_2', 's_conv1_2')
        self.avgpool(self.d_conv1_2, 'pool1')
        self.avgpool(self.s_conv1_2, 'pool1', stream_name='s')

        self.conv2d_relu(self.d_pool1, 5, 'conv2_1')
        self.conv2d_relu(self.d_conv2_1, 7, 'conv2_2')
        self.conv2d_relu(self.s_pool1, 5, 'conv2_1', stream_name='s')
        self.conv2d_relu(self.s_conv2_1, 7, 'conv2_2', stream_name='s')
        self.attention_layer(self.d_conv2_2, self.s_conv2_2, 'd_conv2_2', 's_conv2_2')
        self.avgpool(self.d_conv2_2, 'pool2')
        self.avgpool(self.s_conv2_2, 'pool2', stream_name='s')

        self.conv2d_relu(self.d_pool2, 10, 'conv3_1')
        self.conv2d_relu(self.d_conv3_1, 12, 'conv3_2')
        self.conv2d_relu(self.d_conv3_2, 14, 'conv3_3')
        self.conv2d_relu(self.d_conv3_3, 16, 'conv3_4')
        self.conv2d_relu(self.s_pool2, 10, 'conv3_1', stream_name='s')
        self.conv2d_relu(self.s_conv3_1, 12, 'conv3_2', stream_name='s')
        self.conv2d_relu(self.s_conv3_2, 14, 'conv3_3', stream_name='s')
        self.conv2d_relu(self.s_conv3_3, 16, 'conv3_4', stream_name='s')
        self.attention_layer(self.d_conv3_4, self.s_conv3_4, 'd_conv3_4', 's_conv3_4')
        self.avgpool(self.d_conv3_4, 'pool3')
        self.avgpool(self.s_conv3_4, 'pool3', stream_name='s')

        self.conv2d_relu(self.d_pool3, 19, 'conv4_1')
        self.conv2d_relu(self.d_conv4_1, 21, 'conv4_2')
        self.conv2d_relu(self.d_conv4_2, 23, 'conv4_3')
        self.conv2d_relu(self.d_conv4_3, 25, 'conv4_4')
        self.conv2d_relu(self.s_pool3, 19, 'conv4_1', stream_name='s')
        self.conv2d_relu(self.s_conv4_1, 21, 'conv4_2', stream_name='s')
        self.conv2d_relu(self.s_conv4_2, 23, 'conv4_3', stream_name='s')
        self.conv2d_relu(self.s_conv4_3, 25, 'conv4_4', stream_name='s')
        self.attention_layer(self.d_conv4_4, self.s_conv4_4, 'd_conv4_4', 's_conv4_4')
        self.avgpool(self.d_conv4_4, 'pool4')
        self.avgpool(self.s_conv4_4, 'pool4', stream_name='s')

        self.conv2d_relu(self.d_pool4, 28, 'conv5_1')
        self.conv2d_relu(self.d_conv5_1, 30, 'conv5_2')
        self.conv2d_relu(self.d_conv5_2, 32, 'conv5_3')
        self.conv2d_relu(self.d_conv5_3, 34, 'conv5_4')
        self.conv2d_relu(self.s_pool4, 28, 'conv5_1', stream_name='s')
        self.conv2d_relu(self.s_conv5_1, 30, 'conv5_2', stream_name='s')
        self.conv2d_relu(self.s_conv5_2, 32, 'conv5_3', stream_name='s')
        self.conv2d_relu(self.s_conv5_3, 34, 'conv5_4', stream_name='s')
        self.attention_layer(self.d_conv5_4, self.s_conv5_4, 'd_conv5_4', 's_conv5_4')
        self.avgpool(self.d_conv5_4, 'pool5')
        #self.avgpool(self.s_conv5_4, 'pool5', stream_name='s')

        self.conv2d_relu(self.d_pool5, 37, 'fc6')
        self.conv2d_relu(self.d_fc6, 39, 'fc7')
        #self.regression_output_layer(self.fc7, 'fc8')
        # self.conv2d_relu(self.fc7, 41, 'fc8')
        print("done.")

    def vgg_load(self):
        print("begin to construct single stream vgg......")

        self.conv2d_relu(self.input_diff, 0, 'conv1_1')
        self.conv2d_relu(self.d_conv1_1, 2, 'conv1_2')
        self.avgpool(self.d_conv1_2, 'pool1')

        self.conv2d_relu(self.d_pool1, 5, 'conv2_1')
        self.conv2d_relu(self.d_conv2_1, 7, 'conv2_2')
        self.avgpool(self.d_conv2_2, 'pool2')

        self.conv2d_relu(self.d_pool2, 10, 'conv3_1')
        self.conv2d_relu(self.d_conv3_1, 12, 'conv3_2')
        self.conv2d_relu(self.d_conv3_2, 14, 'conv3_3')
        self.conv2d_relu(self.d_conv3_3, 16, 'conv3_4')
        self.avgpool(self.d_conv3_4, 'pool3')

        self.conv2d_relu(self.d_pool3, 19, 'conv4_1')
        self.conv2d_relu(self.d_conv4_1, 21, 'conv4_2')
        self.conv2d_relu(self.d_conv4_2, 23, 'conv4_3')
        self.conv2d_relu(self.d_conv4_3, 25, 'conv4_4')
        self.avgpool(self.d_conv4_4, 'pool4')

        self.conv2d_relu(self.d_pool4, 28, 'conv5_1')
        self.conv2d_relu(self.d_conv5_1, 30, 'conv5_2')
        self.conv2d_relu(self.d_conv5_2, 32, 'conv5_3')
        self.conv2d_relu(self.d_conv5_3, 34, 'conv5_4')
        self.avgpool(self.d_conv5_4, 'pool5')

        self.conv2d_relu(self.d_pool5, 37, 'fc6')
        self.conv2d_relu(self.d_fc6, 39, 'fc7')
        # self.regression_output_layer(self.fc7, 'fc8')
        # self.conv2d_relu(self.fc7, 41, 'fc8')
        print("done.")


if __name__ == '__main__':
    frame = cv2.imread('frame0.jpg').astype(np.float32)
    diff = cv2.imread('diff0.jpg').astype(np.float32)
    frame = np.expand_dims(frame, 0)
    diff = np.expand_dims(diff, 0)
    model = NnModel(frame, diff)
    model.vgg_load()

