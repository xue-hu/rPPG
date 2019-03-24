#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import tensorflow as tf
import cv2
import two_stream_vgg
import face_vgg
import rnn_model
import process_data
import utils
import math
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ECG_SAMPLE_RATE = 16.0
PLE_SAMPLE_RATE = 256.0
FRAME_RATE = 30.0
N_CLASSES = 2
MODEL = 'reg'  #'classi'


class VideoAnalysis(object):
    def __init__(self, train_video_paths, train_label_paths, train_gt_paths,
                 test_video_paths, test_label_paths, test_gt_paths,
                 batch_size, lr,img_width, img_height):
        self.train_video_paths = train_video_paths
        self.train_label_paths = train_label_paths
        self.train_gt_paths = train_gt_paths
        self.test_video_paths = test_video_paths
        self.test_label_paths = test_label_paths
        self.test_gt_paths = test_gt_paths
        self.width = img_width
        self.height = img_height
        self.duration = 12
        self.lr = lr
        self.sign_loss_weight = 1.0
        self.batch_size = batch_size
        self.gstep = tf.Variable(0, trainable=False, name='global_step')
        self.skip_step = 50000    

    def loading_model(self,network_obj):                     
        self.model = network_obj #two_stream_vgg.TwoStreamVgg(self.batch_size,self.width,self.height)
        self.model.construct_network()
        
        
    def build_graph(self,network_obj):
        self.loading_model(network_obj)
        self.model.inference()
        self.model.loss(MODEL)
        self.model.evaluation(MODEL)
        self.model.optimizer(self.lr,self.gstep)


    def train(self, n_epoch):
        print("begin to train.....")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("initialization completed.")
            writer = tf.summary.FileWriter('./graphs/', sess.graph)
            print("computational graph saved.")
            saver = tf.train.Saver()
            summary_train, summary_test = self.model.create_summary()
            step = self.gstep.eval()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoint_dict/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('loading stored params...........')
            pre_loss = 100
                        
            for epoch in range(n_epoch):
                train_gen = self.model.get_data(self.train_video_paths, self.train_label_paths, self.train_gt_paths,np.arange(2, 601),mode='train')
                step, loss = self.model.train_one_epoch(sess, writer, saver, train_gen,summary_train, epoch, step)   
                #if pre_loss <= loss:
                self.lr = self.lr / 2.0
                pre_loss = loss
                for test_video_path, test_label_path, test_gt_path in zip(self.test_video_paths,self.test_label_paths, self.test_gt_paths):
                    test_gen = self.model.get_data([test_video_path], [test_label_path], [test_gt_path],[1], mode='test')
                    step = self.model.eval_once(sess, writer,test_gen,test_video_path, self.duration, summary_test, epoch, step)                            
            writer.close()


if __name__ == '__main__':
    ############using remote dataset######################################################
    tr_vd_paths = []
    tr_lb_paths = []
    tr_gt_paths = []
    te_vd_paths = []
    te_lb_paths = []
    te_gt_paths = []
    t_id = 1
    for cond in ['lighting']:
        if cond == 'lighting':
            n = [0]
        for i in n:
            tr_vd_path, tr_lb_path = utils.create_file_paths( np.delete(np.arange(1, 27), t_id-1), cond=cond, cond_typ=i)#np.delete(np.arange(1, 27), 11)
            _, tr_gt_path = utils.create_file_paths(np.delete(np.arange(1, 27), t_id-1) , cond=cond, cond_typ=i, sensor_sgn=0)
            te_vd_path, te_lb_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i)
            _, te_gt_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i, sensor_sgn=0)
            tr_vd_paths += tr_vd_path
            tr_lb_paths += tr_lb_path
            tr_gt_paths += tr_gt_path
            te_vd_paths += te_vd_path
            te_lb_paths += te_lb_path
            te_gt_paths += te_gt_path
#     for cond in ['lighting']:
#         if cond == 'lighting':
#             n = [1]
#         for i in n:
#             tr_vd_path, tr_lb_path = utils.create_file_paths(np.delete(np.arange(1, 27), [9,20,22,24   , t_id-1 ]), cond=cond, cond_typ=i)
#             _, tr_gt_path = utils.create_file_paths(np.delete(np.arange(1, 27), [9,20,22,24     , t_id-1 ]), cond=cond, cond_typ=i, sensor_sgn=0)
#             te_vd_path, te_lb_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i)
#             _, te_gt_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i, sensor_sgn=0)
#             tr_vd_paths += tr_vd_path
#             tr_lb_paths += tr_lb_path
#             tr_gt_paths += tr_gt_path
#             te_vd_paths += te_vd_path
#             te_lb_paths += te_lb_path
#             te_gt_paths += te_gt_path
    for cond in ['lighting']:
        if cond == 'lighting':
            n = [3]
        for i in n:
            tr_vd_path, tr_lb_path = utils.create_file_paths(np.delete(np.arange(1, 27), [22   ,t_id-1]), cond=cond, cond_typ=i)
            _, tr_gt_path = utils.create_file_paths(np.delete(np.arange(1, 27), [22   ,t_id-1]), cond=cond, cond_typ=i, sensor_sgn=0)
            te_vd_path, te_lb_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i)
            _, te_gt_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i, sensor_sgn=0)
            tr_vd_paths += tr_vd_path
            tr_lb_paths += tr_lb_path
            tr_gt_paths += tr_gt_path
            te_vd_paths += te_vd_path
            te_lb_paths += te_lb_path
            te_gt_paths += te_gt_path

    for cond in ['movement']:
        if cond == 'movement':
            n = [0]
        for i in n:
            tr_vd_path, tr_lb_path = utils.create_file_paths(np.delete(np.arange(1, 27), [3,4,     t_id-1]), cond=cond, cond_typ=i)
            _, tr_gt_path = utils.create_file_paths(np.delete(np.arange(1, 27), [3,4,      t_id-1]), cond=cond, cond_typ=i, sensor_sgn=0)
            te_vd_path, te_lb_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i)
            _, te_gt_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i, sensor_sgn=0)
            tr_vd_paths += tr_vd_path
            tr_lb_paths += tr_lb_path
            tr_gt_paths += tr_gt_path
            te_vd_paths += te_vd_path
            te_lb_paths += te_lb_path
            te_gt_paths += te_gt_path   

    for cond in ['movement']:
        if cond == 'movement':
            n = [1]
        for i in n:
            tr_vd_path, tr_lb_path = utils.create_file_paths(np.delete(np.arange(1,27), [22     ,t_id-1]), cond=cond, cond_typ=i)
            _, tr_gt_path = utils.create_file_paths(np.delete(np.arange(1, 27), [22     ,t_id-1]), cond=cond, cond_typ=i, sensor_sgn=0)
            te_vd_path, te_lb_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i)
            _, te_gt_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i, sensor_sgn=0)
            tr_vd_paths += tr_vd_path
            tr_lb_paths += tr_lb_path
            tr_gt_paths += tr_gt_path
            te_vd_paths += te_vd_path
            te_lb_paths += te_lb_path
            te_gt_paths += te_gt_path   

    tr_vd_path, tr_lb_path = utils.create_extra_file_paths(np.delete(np.arange(1,16),[7,8,11]))
    tr_vd_paths += tr_vd_path
    tr_lb_paths += tr_lb_path
    tr_gt_paths += tr_lb_path
    


    model = VideoAnalysis(tr_vd_paths, tr_lb_paths, tr_gt_paths, te_vd_paths, te_lb_paths, te_gt_paths, batch_size=64 ,lr=0.08, img_height=112, img_width=112 )
    window_size = 5
    network = rnn_model.RnnModel(model.batch_size,window_size,model.width,model.height)
    model.build_graph(network)
    model.train(20)
