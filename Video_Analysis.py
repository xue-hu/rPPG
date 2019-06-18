#!/usr/bin/python3.5
__author__ = 'Iris'

import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import cv2
import random
import two_stream_vgg
import face_vgg
import matrix_completion
import rnn_model
import SegNet
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
                 batch_size,img_width, img_height):
        self.train_video_paths = train_video_paths
        self.train_label_paths = train_label_paths
        self.train_gt_paths = train_gt_paths
        self.test_video_paths = test_video_paths
        self.test_label_paths = test_label_paths
        self.test_gt_paths = test_gt_paths
        self.width = img_width
        self.height = img_height
        self.duration = 12
        self.sign_loss_weight = 1.0
        self.batch_size = batch_size
        self.gstep = tf.Variable(0, trainable=False, name='global_step')
        self.skip_step = 50000    

    def loading_model(self,network_obj):                     
        self.model = network_obj 
        self.model.construct_network()
        
    def get_train_batch(self,clips):
        train_gen = self.model.get_data(self.train_video_paths,
                                    self.train_label_paths,
                                    self.train_gt_paths,
                                    clips,mode='train')
        feat_train = []
        try:
            while True:
                norm_batch, seq_batch, m_batch,lb_batch,gt_batch = next(train_gen)
                feat_train.append((norm_batch,seq_batch, m_batch, lb_batch, gt_batch))
        except StopIteration:
            pass
        return feat_train
        
    def build_graph(self,network_obj):
        self.loading_model(network_obj)
        self.model.inference()
        self.model.loss(MODEL)
        self.model.evaluation(MODEL)
        self.model.optimizer(self.gstep)
        self.model.segnet.optimizer(self.gstep)


    def train(self, n_epoch):
        print("begin to train.....")
        with tf.Session(config=config) as sess: 
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
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
            prob_li = np.arange(2,121) 
            random.shuffle(prob_li)
            tr_step = math.ceil(len(prob_li)/20.)
            for epoch in range(0,n_epoch,20):
                for i in range(20):
                    train_data = self.get_train_batch( prob_li[i*tr_step:(i+1)*tr_step])
                    step, loss = self.model.train_one_epoch(sess, writer, saver,
                                            train_data,summary_train, 
                                            (epoch+i), step)                
                    self.model.lr = self.model.lr * 0.95
                    pre_loss = loss
#                     saver.save(sess, './checkpoint_dict/', global_step=step)
                    if (epoch+i+1)%2 == 0 and (epoch+i) > 12:
                        for test_video_path, test_label_path, test_gt_path in zip(self.test_video_paths,self.test_label_paths, self.test_gt_paths):
                            test_gen = self.model.get_data([test_video_path], [test_label_path], [test_gt_path],[1], mode='test')
                            step = self.model.eval_once(sess, writer,test_gen,test_video_path, self.duration, summary_test, (epoch+i), step) 

            writer.close()


if __name__ == '__main__':
    ############using remote dataset######################################################
    tr_vd_paths = []
    tr_lb_paths = []
    tr_gt_paths = []
    te_vd_paths = []
    te_lb_paths = []
    te_gt_paths = []
    t_id = 3
    for cond in ['lighting']:
        if cond == 'lighting':
            n = [0]
        for i in n:
            tr_vd_path, tr_lb_path = utils.create_file_paths( np.delete(np.arange(1, 27),[3,t_id-1]), cond=cond, cond_typ=i)#np.delete(np.arange(1, 27), 11)
            _, tr_gt_path = utils.create_file_paths(np.delete(np.arange(1, 27),[3,t_id-1]) , cond=cond, cond_typ=i, sensor_sgn=0)
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
#             tr_vd_path, tr_lb_path = utils.create_file_paths(np.delete(np.arange(1, 27), [9,15,20,22,24,t_id-1   ]), cond=cond, cond_typ=i)
#             _, tr_gt_path = utils.create_file_paths(np.delete(np.arange(1, 27), [9,15,20,22,24 ,t_id-1     ]), cond=cond, cond_typ=i, sensor_sgn=0)
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
            tr_vd_path, tr_lb_path = utils.create_file_paths(np.delete(np.arange(1, 27), [3,22,t_id-1  ]), cond=cond, cond_typ=i)
            _, tr_gt_path = utils.create_file_paths(np.delete(np.arange(1, 27), [3,22,t_id-1   ]), cond=cond, cond_typ=i, sensor_sgn=0)
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
            tr_vd_path, tr_lb_path = utils.create_file_paths(np.delete(np.arange(1, 27), [3,21,t_id-1]), cond=cond, cond_typ=i)
            _, tr_gt_path = utils.create_file_paths(np.delete(np.arange(1, 27), [3,21,t_id-1]), cond=cond, cond_typ=i, sensor_sgn=0)
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
            tr_vd_path, tr_lb_path = utils.create_file_paths(np.delete(np.arange(1,27), [3,22 ,t_id-1   ]), cond=cond, cond_typ=i)#  ,t_id-1
            _, tr_gt_path = utils.create_file_paths(np.delete(np.arange(1, 27), [3,22 ,t_id-1    ]), cond=cond, cond_typ=i, sensor_sgn=0)
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
            n = [2]
        for i in n:
            tr_vd_path, tr_lb_path = utils.create_file_paths(np.delete(np.arange(1,27), [t_id-1   ]), cond=cond, cond_typ=i)#  1,3,4,7,9,11,12,13,14,17,21,22 ,
            _, tr_gt_path = utils.create_file_paths(np.delete(np.arange(1, 27), [t_id-1    ]), cond=cond, cond_typ=i, sensor_sgn=0)#1,3,4,7,9,11,12,13,14,17,21,22 ,
            te_vd_path, te_lb_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i)
            _, te_gt_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i, sensor_sgn=0)
            tr_vd_paths += tr_vd_path
            tr_lb_paths += tr_lb_path
            tr_gt_paths += tr_gt_path
            te_vd_paths += te_vd_path
            te_lb_paths += te_lb_path
            te_gt_paths += te_gt_path   

#     tr_vd_path, tr_lb_path = utils.create_extra_file_paths(np.delete(np.arange(1,16),[7,8,11]))#np.delete(np.arange(1,16),[7,8,11])
#     tr_vd_paths += tr_vd_path
#     tr_lb_paths += tr_lb_path
#     tr_gt_paths += tr_lb_path
    


    model = VideoAnalysis(tr_vd_paths, tr_lb_paths, tr_gt_paths, te_vd_paths, te_lb_paths, te_gt_paths, batch_size=1,img_height=32, img_width=32 )
    window_size = 45
#     network = rnn_model.RnnModel(0.1,model.batch_size,window_size,model.width,model.height)
#     network = face_vgg.FaceVgg(0.1,model.batch_size,window_size,model.width,model.height)
    #network = two_stream_vgg.TwoStreamVgg(model.batch_size,model.width,model.height)
    
    network = matrix_completion.MatComplt(0.1,model.batch_size,window_size,model.width,model.height)
#     network = SegNet.SegNet(0.1,model.batch_size,window_size,model.width,model.height)

    model.build_graph(network)
    model.train(2000)
