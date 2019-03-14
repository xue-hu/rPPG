#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import tensorflow as tf
import utils
import scipy.io
import numpy as np
import cv2
import cnn_model


class RnnModel(object):
    def __init__(self, fr_seq, keep_prob):
        self.fr_seq = fr_seq    
        self.temp = tf.constant(1.5)
        self.hidden_sizes = [128, 256]
        self.batch_size = fr_seq.shape[0]
        self.length = 3
        self.lr = 0.0003
        self.skip_step = 1
        self.num_steps = 3 # for RNN unrolled
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
    def create_feat_seq():
        frame = self.fr_seq
        self.feature_extracter = cnn_model.CnnModel(frame, frame, 1)
        self.feature_extracter.vgg_face_load(frame)
        
        
    def create_rnn(self, feat_seq):
        layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_sizes]
        cells = tf.nn.rnn_cell.MultiRNNCell(layers)
        zero_states = cells.zero_state(self.batch_size, dtype=tf.float32)
        self.init_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]]) 
                                for state in zero_states])
        self.output, self.out_state = tf.nn.dynamic_rnn(cells, feat_seq, self.length, self.init_state)

    def create_model(self):
        self.create_feat_seq()
        self.create_rnn(self.self.feature_extracter.fc7)               
        
        self.logits = tf.layers.dense(self.output, 1, None)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[:, :-1], 
                                                        labels=seq[:, 1:])
        self.loss = tf.reduce_sum(loss)
        self.sample = tf.multinomial(tf.exp(self.logits[:, -1] / self.temp), 1)[:, 0] 
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)
        