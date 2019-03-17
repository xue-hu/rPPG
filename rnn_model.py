#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import tensorflow as tf
import utils
import scipy.io
import numpy as np
import cv2
import face_vgg
import process_data


class RnnModel(object):
    def __init__(self,batch_size,length):
        self.batch_size = batch_size
        self.length = length
        self.temp = tf.constant(1.5)
        self.hidden_sizes = [128, 256]
        self.lr = 0.0003
        self.skip_step = 1
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob', shape=[])
            
        
    def create_feat_seq(self,height, width ):
        self.feature_extracter = face_vgg.FaceVgg(self.batch_size*self.length)
        self.feature_extracter.construct_network(height, width) 
        self.feature_seq = tf.reshape(self.feature_extracter.fc7,(self.batch_size, self.length, -1))       

        
    def create_rnn(self):
        layers = [tf.nn.rnn_cell.GRUCell(size) for size in self.hidden_sizes]
        cells = tf.nn.rnn_cell.MultiRNNCell(layers)
        cells = tf.nn.rnn_cell.DropoutWrapper(cells,output_keep_prob=self.keep_prob)

        zero_states = cells.zero_state(self.batch_size, dtype=tf.float32)
        self.init_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]]) 
                                for state in zero_states])
        seq_length = tf.convert_to_tensor([self.length]*self.batch_size, dtype=tf.int32)
        self.output, self.out_state = tf.nn.dynamic_rnn(cell=cells, 
                                        inputs=self.feature_seq, 
                                        sequence_length=seq_length, 
                                        initial_state=self.init_state,
                                        dtype=tf.float32)

    def create_model(self,height, width):
        self.create_feat_seq(height, width)
        self.create_rnn()               
        
        
#         self.logits = tf.layers.dense(self.output, 1, None)
#         loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[:, :-1], 
#                                                         labels=seq[:, 1:])
#         self.loss = tf.reduce_sum(loss)
#         self.sample = tf.multinomial(tf.exp(self.logits[:, -1] / self.temp), 1)[:, 0] 
#         self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)
        
if __name__ == '__main__':
    tr_vd_paths = []
    tr_lb_paths = []
    tr_gt_paths = []
    te_vd_paths = []
    te_lb_paths = []
    te_gt_paths = []
    window_size = 2
    batch_size = 64
    window_size = 2
    height = 112
    width = 112
    
    for cond in ['lighting']:
        if cond == 'lighting':
            n = [0]
        for i in n:
            tr_vd_path, tr_lb_path = utils.create_file_paths(np.arange(1, 2), cond=cond, cond_typ=i)
            _, tr_gt_path = utils.create_file_paths(np.arange(1, 2) , cond=cond, cond_typ=i, sensor_sgn=0)
            te_vd_path, te_lb_path = utils.create_file_paths([1], cond=cond, cond_typ=i)
            _, te_gt_path = utils.create_file_paths([1], cond=cond, cond_typ=i, sensor_sgn=0)
            tr_vd_paths += tr_vd_path
            tr_lb_paths += tr_lb_path
            tr_gt_paths += tr_gt_path
            te_vd_paths += te_vd_path
            te_lb_paths += te_lb_path
            te_gt_paths += te_gt_path
    
    rnn_model = RnnModel(batch_size,window_size)
    rnn_model.create_model(height, width)
    seq_batch_gen = rnn_model.get_data(tr_vd_paths, tr_lb_paths, tr_gt_paths, window_size, batch_size, width=112, height=112)  
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("initialization completed.")
          
        try:
            while True:
                seq_batch, lb_batch, gt_batch = next(seq_batch_gen)
                seq_batch = np.reshape(seq_batch,[batch_size*window_size, height, width, 3])
                feat_seq, output = sess.run([rnn_model.feature_seq, rnn_model.output], feed_dict={rnn_model.feature_extracter.input_img:seq_batch,rnn_model.keep_prob:1,rnn_model.feature_extracter.keep_prob:1
                                                                                                 })
                print(np.asarray(feat_seq).shape)
                print(np.asarray(output).shape)
                
        except StopIteration:
                    pass
