#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import time
import tensorflow as tf
import utils
import scipy.io
import numpy as np
import math
import cv2
import face_vgg
import fcn
import process_data

FRAME_RATE = 30.0

class RnnModel(object):
    def __init__(self,batch_size,length,height, width ):
        self.batch_size = batch_size
        self.length = length
        self.height = height
        self.width = width
        self.temp = tf.constant(1.5)
        self.hidden_sizes = [128, 256]
        self.lr = 0.0003
        self.skip_step = 1
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob', shape=[])
            
        
    def create_feat_seq(self):
        self.feature_extracter = face_vgg.FaceVgg(self.batch_size*self.length,self.height, self.width, model='face_vgg')
        #self.feature_extracter = fcn.FCN(self.batch_size*self.length,self.height, self.width, model='fcn')
        self.feature_extracter.construct_network() 
        self.feature_extracter.inference()
        with tf.variable_scope('input_seq', reuse=tf.AUTO_REUSE) as scope:
            self.feature_seq = tf.reshape(self.feature_extracter.output,(self.batch_size, self.length, -1))       

        
    def create_rnn(self):
        with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE) as scope:    
            layers = [tf.nn.rnn_cell.GRUCell(size,kernel_initializer=tf.random_normal_initializer(stddev=0.4)) for size in self.hidden_sizes]
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

    def construct_network(self):
        print("begin to construct LT-RCN......")
        self.create_feat_seq()
        self.create_rnn() 
        
        
    def get_data(self, tr_vd_paths, tr_lb_paths, tr_gt_paths,clips, mode='train'):
        return self.feature_extracter.get_data(tr_vd_paths, tr_lb_paths, tr_gt_paths, self.length,clips=clips,mode=mode)
        
        
    def inference(self):
        with tf.name_scope('final_output'):
            self.output = tf.layers.dense(self.output, 1,kernel_initializer=tf.random_normal_initializer(stddev=0.4))
        with tf.name_scope('predictions'):
            self.pred = tf.reshape(self.output,shape=[self.batch_size,self.length])
        
    def loss(self,model=None):
        with tf.name_scope('labels'):
            self.labels = tf.placeholder(dtype=tf.float32, name='ppg', shape=[self.batch_size,self.length])
        print("crteate loss-Function.....")
        with tf.name_scope('loss'):            
            self.loss = tf.losses.mean_squared_error(self.pred, self.labels)
            #self.loss = tf.losses.huber_loss(self.logits, self.labels)# + self.sign_loss_weight*self.sign_loss  
    
    def optimizer(self,lr,gstep):
        print("crteate optimizer.....")
        optimizer = tf.train.AdadeltaOptimizer(lr)
#         grads, variable_names = zip(*optimizer.compute_gradients(self.loss))
#         clip_grads,_ = tf.clip_by_global_norm(grads, 500)
#         self.grads = [(g, v) for g,v in zip(clip_grads, variable_names)]
#         self.grads_norm = tf.global_norm([g[0] for g in self.grads])
#         self.opt = optimizer.apply_gradients(self.grads, global_step=self.gstep)
        self.opt = optimizer.minimize(self.loss, global_step=gstep)
        
        
    def evaluation(self, model=None):
        print("create evaluation methods.....")        
        self.hrs = tf.placeholder(dtype=tf.float32, name='pred_hr', shape=[self.batch_size, ])
        self.gts = tf.placeholder(dtype=tf.float32, name='ground_truth', shape=[self.batch_size, ])
        with tf.name_scope('hr_accuracy'):
            diff = tf.abs(self.hrs - self.gts)
            indicator = tf.where(diff < 3)
            total_match = tf.cast(tf.size(indicator), tf.float32)
            self.hr_accuracy = tf.truediv(total_match, tf.cast(self.batch_size, tf.float32))   
        with tf.name_scope('hr_MAE'):                    
            self.hr_mae = tf.keras.metrics.mean_absolute_error(self.gts, self.hrs)
        
        
    def create_summary(self):
        print("crteate summary.....")
        summary_loss = tf.summary.scalar('loss', self.loss) 
        summary_hr_accuracy = tf.summary.scalar('hr_accuracy', self.hr_accuracy)
        summary_hr_mae = tf.summary.scalar('hr_mae', self.hr_mae)
        #summary_norm = tf.summary.scalar('grads_norm', self.grads_norm)
        #summary_grad = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in self.grads])
        summary_train = tf.summary.merge([summary_loss])#, summary_grad, summary_norm]) 
        summary_test = tf.summary.merge([summary_loss, summary_hr_accuracy, summary_hr_mae]) 
        return summary_train, summary_test
        
        
    def train_one_epoch(self, sess, writer, saver, train_gen,summary_op, epoch, step):
        total_loss = 0
        n_batch = 1
        start_time = time.time()        
        try:
            while True:
                print("epoch " + str(epoch + 1) + "-" + str(n_batch))
                seq_batch,lb_batch, gt_batch = next(train_gen)
                seq_batch = np.reshape(np.asarray(seq_batch),[self.batch_size*self.length, self.height, self.width, 3])
                pred,loss,labels,_,summary = sess.run([self.pred,
                                     self.loss,
                                     self.labels,
                                     self.opt,
                                     summary_op], 
                                     feed_dict={self.feature_extracter.input_img:seq_batch,
                                                   self.labels:lb_batch,
                                                   self.keep_prob:0.95,
                                                   self.feature_extracter.keep_prob:0.95
                                                                                                 })
                total_loss += loss
                print('label:')
                print(labels[0,-2:])
                print('pred:')
                print(pred[0,-2:])
                if n_batch % 3000 == 0 :
                    saver.save(sess, './checkpoint_dict/', global_step=step)
                writer.add_summary(summary, global_step=step)
                step += 1
                print('Average loss at batch {0}: {1}'.format(n_batch, total_loss / n_batch))
                n_batch += 1
                
        except StopIteration:
            pass
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batch))
        print('Took:{0} seconds'.format(time.time() - start_time))
        return step,(total_loss / n_batch)

        
        
    def eval_once(self, sess, writer, test_gen,test_video_path, duration, summary_op, epoch, step):
        print("begin to evaluate.....")        
        thd = math.ceil(duration * FRAME_RATE / self.batch_size) + 1
        path = test_video_path.split('/')
        prob_id = path[4]
        cond = path[5].split('_')[0]
        if not os.path.exists('./predict_results/'):
            os.makedirs('./predict_results/')
        fileObject = open('./predict_results/'+cond+'-'+prob_id+'-ppg('+str(epoch)+').txt', 'w')
        fileObject.write('\t'*200+'\n') 
        
        gt_accuracy = 0
        ref_accuracy = 0
        n_test = 0
        hr_li = []
        gt_li = []
        ref_li = []
        ppgs = []
        ref_ppgs = []
        try:
            while True:
                seq_batch,lb_batch, gt_batch = next(test_gen)
                seq_batch = np.reshape(np.asarray(seq_batch),[self.batch_size*self.length, self.height, self.width, 3])
                gt_batch = np.asarray(gt_batch, dtype=np.float32)[:,-1]
                pred,labels,_ = sess.run([self.pred,self.labels, self.loss], 
                                     feed_dict={self.feature_extracter.input_img:seq_batch,
                                                   self.labels:lb_batch,
                                                   self.keep_prob:1,
                                                   self.feature_extracter.keep_prob:1})
                

                print('label:')
 
                labels = np.asarray(labels, dtype=np.float32)[:,-1]
                print(labels[:2])                  
                pred = pred
                pred = np.asarray(pred, dtype=np.float32)[:,-1]
                print('pred:')
                print(pred[:2])
                ppgs += pred.tolist()


                ref_ppgs += labels.tolist()
                n_test += 1                     
                if n_test >= thd:
                    print('cvt ppg >>>>>>>>>>>>')
                    hrs = process_data.get_hr(ppgs, self.batch_size,duration, fs=FRAME_RATE)
                    accuracy, summary = sess.run([self.hr_accuracy, summary_op], feed_dict={ 
                        self.feature_extracter.input_img:seq_batch,
                        self.labels:lb_batch,
                        self.hrs: hrs,
                        self.gts: gt_batch,
                        self.keep_prob:1,
                        self.feature_extracter.keep_prob:1})
                    gt_accuracy += accuracy

                    ref_hrs = process_data.get_hr(ref_ppgs, self.batch_size, duration, fs=FRAME_RATE)
                    accuracy = sess.run([self.hr_accuracy], feed_dict={
                        self.feature_extracter.input_img:seq_batch,
                        self.labels:lb_batch,
                        self.hrs: hrs,
                        self.gts: ref_hrs,
                        self.keep_prob:1,
                        self.feature_extracter.keep_prob:1})[0]
                    ref_accuracy += accuracy

                    for hr, gt, ref_hr, i, j in zip(hrs, gt_batch, ref_hrs, pred, labels):
                        hr_li.append(hr)
                        gt_li.append(gt)
                        ref_li.append(ref_hr)
                        fileObject.write(str(round(i,5))+'      '+str(round(j,5))+'    :    '+str(int(hr))+'    '+ str(int(gt))+'    '+ str(int(ref_hr))+'\n')  

                    writer.add_summary(summary, global_step=step)
                    step += 1                       
        except StopIteration:
            pass
        mae = (np.abs(np.asarray(hr_li) - np.asarray(gt_li))).mean(axis=None)
        ref_mae = (np.abs(np.asarray(hr_li) - np.asarray(ref_li))).mean(axis=None)
        fileObject.seek(0)
        fileObject.write( 'gt_accuracy: '+str(gt_accuracy / (n_test-thd+1))+'\n' ) 
        fileObject.write('MAE: '+str(mae)+'\n')  
        fileObject.write('ref_accuracy: '+str(ref_accuracy / (n_test-thd+1))+'\n') 
        fileObject.write('ref_MAE: '+str(ref_mae)+'\n')  
        fileObject.close() 
        print('################Accuracy at epoch {0}: {1}'.format(epoch, gt_accuracy / ( n_test - thd) ))
        return step



        
# if __name__ == '__main__':
#     tr_vd_paths = []
#     tr_lb_paths = []
#     tr_gt_paths = []
#     te_vd_paths = []
#     te_lb_paths = []
#     te_gt_paths = []
#     window_size = 2
#     batch_size = 64
#     window_size = 2
#     height = 112
#     width = 112
    
#     for cond in ['lighting']:
#         if cond == 'lighting':
#             n = [0]
#         for i in n:
#             tr_vd_path, tr_lb_path = utils.create_file_paths(np.arange(1, 2), cond=cond, cond_typ=i)
#             _, tr_gt_path = utils.create_file_paths(np.arange(1, 2) , cond=cond, cond_typ=i, sensor_sgn=0)
#             te_vd_path, te_lb_path = utils.create_file_paths([1], cond=cond, cond_typ=i)
#             _, te_gt_path = utils.create_file_paths([1], cond=cond, cond_typ=i, sensor_sgn=0)
#             tr_vd_paths += tr_vd_path
#             tr_lb_paths += tr_lb_path
#             tr_gt_paths += tr_gt_path
#             te_vd_paths += te_vd_path
#             te_lb_paths += te_lb_path
#             te_gt_paths += te_gt_path
    
#     rnn_model = RnnModel(batch_size,window_size,height, width)
#     rnn_model.construct_network()
#     seq_batch_gen = rnn_model.get_data(tr_vd_paths, tr_lb_paths, tr_gt_paths)  
#     rnn_model.inference()
#     rnn_model.loss()
    
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print("initialization completed.")
          
#         try:
#             while True:
#                 seq_batch, lb_batch, gt_batch = next(seq_batch_gen)
#                 seq_batch = np.reshape(np.asarray(seq_batch),[batch_size*window_size, height, width, 3])
#                 pred, output, loss = sess.run([rnn_model.pred, rnn_model.output,rnn_model.loss], 
#                                          feed_dict={rnn_model.feature_extracter.input_img:seq_batch,
#                                                    rnn_model.labels:lb_batch,
#                                                    rnn_model.keep_prob:1,
#                                                    rnn_model.feature_extracter.keep_prob:1
#                                                                                                  })
#                 print(np.asarray(pred).shape)
#                 print(np.asarray(lb_batch).shape)
#                 print(loss)
                
#         except StopIteration:
#                     pass
