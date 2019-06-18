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
import SegNet
import process_data

# VGG-19 parameters file
N_CLASSES = 2
FRAME_RATE = 30.0


class MatComplt(cnn_model.CnnModel): 
    
    def get_data(self, video_paths, label_paths, gt_paths,clips,mode):
        print("create generator....")   
        batch_gen = process_data.get_seq_batch(video_paths, label_paths, gt_paths,
                                               clips=clips,batch_size = self.batch_size,
                                               window_size = self.length,
                                               height=self.height,width=self.width,
                                               mode=mode)       
        return batch_gen
    
    
    def construct_network(self):
        print("begin to construct Auto-encoder......")
        self.segnet = SegNet.SegNet(0.1,self.batch_size*self.length,1,self.width,self.height)
        self.segnet.build_graph()
        with tf.name_scope('Input'):
            seg_map = tf.reshape(self.segnet.pred_mask,shape=[self.batch_size*self.length,self.height,self.width,1])
            self.input_nn = tf.multiply(self.segnet.input_imgs,seg_map)
            input_n = self.globalpool(self.input_nn,activation='avg')
            self.input_n = tf.reshape(input_n,shape=[self.batch_size*self.length,3])
            
            
#         with tf.name_scope('Input'):   
#             self.input_norm = tf.placeholder(dtype=tf.float32, name='input_norm',
#                                             shape=[self.batch_size,self.length,3])
#             self.input_n = tf.reshape(self.input_norm,
#                                             shape=[self.batch_size*self.length,3])
           
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE,regularizer=tf.contrib.layers.l2_regularizer(0.01)) as scope:

###############Reconstruct ICA########################################################################################                  
            encode_1 = tf.layers.dense( self.input_n, 1024,activation=tf.nn.elu,
                                                         kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='encode_1')
            encode_1 = tf.nn.dropout( encode_1, self.keep_prob)                     
                   
            encode_2 = tf.layers.dense(encode_1,512,activation=tf.nn.elu,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='encode_2')
            encode_2 = tf.nn.dropout( encode_2, self.keep_prob)
            
            encode_3 = tf.layers.dense(encode_2,256,activation=tf.nn.elu,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='encode_3')
            encode_3 = tf.nn.dropout( encode_3, self.keep_prob)

            encode = tf.layers.dense( encode_3,3,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='encode_0')

#         with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE,regularizer=tf.contrib.layers.l2_regularizer(0.01)) as scope:
            
            decode_3 = tf.layers.dense(encode,256,activation=tf.nn.elu,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='decode_3')
            decode_3 = tf.nn.dropout( decode_3, self.keep_prob)
            
            decode_2 = tf.layers.dense(decode_3,512,activation=tf.nn.elu,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='decode_2')
            decode_2 = tf.nn.dropout( decode_2, self.keep_prob)

            decode_1 = tf.layers.dense( decode_2, 1024, activation=tf.nn.elu,
                             kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='decode_1')
            decode_1 = tf.nn.dropout( decode_1, self.keep_prob)
            
            decode = tf.layers.dense( decode_1, 3, activation=tf.nn.tanh,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1),name='decode_0')

        setattr(self, 'encode', encode)
        setattr(self, 'decode', decode)
        setattr(self,'activ',[encode])
           
       
    def inference(self):        
        with tf.variable_scope('predictions'):
            self.output = tf.layers.dense(self.encode,1,
                            kernel_initializer=tf.random_normal_initializer(stddev=0.1))
            self.output = tf.reshape(self.output, shape=[self.batch_size*self.length,])        
            self.pred = tf.reshape(self.output, shape=[self.batch_size*self.length,])

            self.origin = tf.reshape(self.decode, shape=[self.batch_size*self.length, 3])

                            
    def loss(self,model=None):
        with tf.variable_scope('labels'):
            self.labels = tf.placeholder(dtype=tf.float32, name='ppg_diff', shape=[self.batch_size,self.length])
            self.label = tf.reshape(self.labels, shape=[self.batch_size*self.length,])
        print("crteate loss-Function.....")
        with tf.name_scope('loss'):
            sum_k1 = tf.convert_to_tensor([tf.reduce_sum(tf.nn.tanh(p)) for p in self.activ],dtype=np.float32)
            self.l1_loss = tf.reduce_sum(sum_k1)
            self.code_loss = tf.losses.mean_squared_error(self.input_n, self.origin)
            self.reg_loss = tf.losses.mean_squared_error(self.output, self.label) 
            w_regulation = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = self.reg_loss + self.code_loss + tf.reduce_sum(w_regulation) #+ 0.1 * self.l1_loss
            
            
    def optimizer(self,gstep):
        print("crteate optimizer.....")
        with tf.control_dependencies( tf.get_collection( tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
            
            
#             tvars = tf.trainable_variables()
#             reg_vars = [var for var in tvars if 'predictions' in var.name]
#             code_vars = [var for var in tvars if not 'predictions' in var.name]
            
#             self.reg_opt = optimizer.minimize(self.reg_loss,var_list=reg_vars, 
#                                               global_step=gstep)
            
#             w_regulation = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#             self.code_opt = optimizer.minimize(self.code_loss+ tf.reduce_sum(w_regulation), 
#                                                var_list=code_vars, 
#                                                global_step=gstep)
            self.opt = optimizer.minimize(self.loss,global_step=gstep)
#         grads, variable_names = zip(*optimizer.compute_gradients(self.loss))
#         clip_grads,_ = tf.clip_by_global_norm(grads, 20)
#         self.grads = [(g, v) for g,v in zip(clip_grads, variable_names)]
#         self.grads_norm = tf.global_norm([g[0] for g in self.grads])
#         self.opt = optimizer.apply_gradients(self.grads, global_step=gstep)
            
    
    
    def evaluation(self, model=None):
        print("create evaluation methods.....")
        self.hrs = tf.placeholder(dtype=tf.float32, name='pred_hr', shape=[self.batch_size*self.length, ])
        self.gts = tf.placeholder(dtype=tf.float32, name='ground_truth', shape=[self.batch_size*self.length, ])
        with tf.variable_scope('hr_accuracy'):            
            diff = tf.abs(self.hrs - self.gts)
            indicator = tf.where(diff < 3)
            total_match = tf.cast(tf.size(indicator), tf.float32)
            self.hr_accuracy = tf.truediv(total_match, tf.cast(self.batch_size*self.length, tf.float32))   
        with tf.name_scope('hr_MAE'):                    
            self.hr_mae = tf.keras.metrics.mean_absolute_error(self.gts, self.hrs)
           

    def create_summary(self):
        print("crteate summary.....")
        summary_loss = tf.summary.scalar('reg_loss', self.reg_loss) 
        summary_code_loss = tf.summary.scalar('code_loss', self.code_loss) 
        summary_hr_mae = tf.summary.scalar('hr_mae', self.hr_mae)
#         summary_norm = tf.summary.scalar('grads_norm', self.grads_norm)
#         summary_grad = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in self.grads])
        summary_train = tf.summary.merge([summary_code_loss,summary_loss])#, summary_grad, summary_norm]) 
        summary_test = tf.summary.merge([summary_code_loss,summary_loss,summary_hr_mae]) 
        return summary_train, summary_test
    
    
    def train_one_epoch(self, sess, writer, saver, train_data,summary_op, epoch, step):
        total_loss = 0
        n_batch = 1
        start_time = time.time()   
#         if epoch % 2 != 0 :
#             opt = self.code_opt
#             print('............encoder..................')
#         else:
#             opt = self.reg_opt
#             print('............regression..................')
        for (norm_frame_batch,frame_batch, m_batch,lb_batch, gt_batch) in train_data:
            print("epoch " + str(epoch + 1) + "-" + str(n_batch))               
            lb_batch = np.asarray(lb_batch, dtype=np.float32)
            gt_batch = np.asarray(gt_batch, dtype=np.float32)
            frame_batch = np.reshape(np.asarray(frame_batch, dtype=np.float32),[self.batch_size*self.length,self.height,self.width,3])
            norm_frame_batch = np.asarray(norm_frame_batch, dtype=np.float32)
            m_batch = np.reshape(np.asarray(m_batch, dtype=np.float32),[self.batch_size*self.length,self.height,self.width,1])
            if epoch > 12:
                pred,loss,_,labels,_,summary,_,_ = sess.run([self.pred,
                                     self.reg_loss,
                                     self.code_loss,
                                     self.label,
                                     self.opt,                    
                                     summary_op,
                                     self.segnet.pred,
                                     self.segnet.loss],
                                     feed_dict={self.segnet.input_imgs:frame_batch,
                                            self.segnet.mask:m_batch,
                                            self.labels:lb_batch,
                                            self.is_training: True,
                                            self.segnet.is_training: True,
                                            self.segnet.keep_prob:1,
                                            self.keep_prob:1  })
    #                                  feed_dict={self.input_imgs:frame_batch,
    #                                             self.input_norm:norm_frame_batch,
    #                                                self.labels:lb_batch,
    #                                             self.is_training: True,
    #                                                self.keep_prob:1  })
            if epoch <= 12:
                pred,loss,_,labels,summary,_,_ = sess.run([self.pred,self.reg_loss,self.code_loss,self.label,summary_op,
                                self.segnet.opt,self.segnet.update_op],
                                         feed_dict={self.segnet.input_imgs:frame_batch,
                                                self.segnet.mask:m_batch,
                                                self.labels:lb_batch,
                                                self.is_training: True,
                                                self.segnet.is_training: True,
                                                self.segnet.keep_prob:1,
                                                self.keep_prob:1})
            total_loss += loss
            print('label:')
            print(labels[:2])
            print('pred:')
            print(pred[:2])
            writer.add_summary(summary, global_step=step)
            step += 1
            print('Average loss at batch {0}: {1}'.format(n_batch, total_loss / n_batch))
            n_batch += 1      
#####################################################################################################################
            if (n_batch-2)%50 == 0:                   
                pred_mask,re = sess.run([self.segnet.pred_mask,self.input_nn],
                                      feed_dict={self.segnet.input_imgs:frame_batch,
                                            self.segnet.mask:m_batch,
                                               self.segnet.is_training: True,
                                               self.segnet.keep_prob:1  })
                if not os.path.exists('./pred_mask/'):
                    os.makedirs('./pred_mask/')
                pred_mask = np.asarray(pred_mask ,dtype=np.float32).reshape((self.batch_size*self.length,self.height,self.width,1)) 
                pred_mask = pred_mask[0,:,:,:] * 255
                pred_mask[pred_mask > 255] = 255
#                 pred_mask = pred_mask.astype(np.uint8)
#                 re = cv2.bitwise_and(frame_batch[0,:,:,:],frame_batch[0,:,:,:],mask = pred_mask[0,:,:,:])
                print('############################################################################')
                cv2.imwrite( ('./pred_mask/mask-'+str(epoch)+'-'+str(step) + '.jpg'),pred_mask)
#                 cv2.imwrite( ('./pred_mask/re-'+str(epoch)+'-'+str(step) + '.jpg'),np.asarray(re)[0,:,:,:])
##########################################################################################################################
        
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batch))
        print('Took:{0} seconds'.format(time.time() - start_time))
        return step,(total_loss / n_batch)

        
        
    def eval_once(self, sess, writer, test_gen,test_video_path, duration, summary_op, epoch, step):
        print("begin to evaluate.....")        
        thd = math.ceil(duration * FRAME_RATE / (self.batch_size*self.length)) + 1
        path = test_video_path.split('/')
        prob_id = path[4]
        cond = path[5].split('_')[0]
        if not os.path.exists('./predict_results/'):
            os.makedirs('./predict_results/')
        fileObject = open('./predict_results/'+cond+'-'+prob_id+'-ppg('+str(epoch)+').txt', 'w')
        sumfile = open('./predict_results/'+prob_id+'-summary.txt', 'a')
        fileObject.write('\t'*200+'\n') 
#         ff = open('./predict_results/'+cond+'-'+prob_id+'-bss('+str(epoch)+').txt', 'w')
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
                norm_frame_batch,frame_batch,_,lb_batch,gt_batch = next(test_gen)               
                gt_batch = np.asarray(gt_batch, dtype=np.float32).flatten()#[:,-1]
                lb_batch = np.asarray(lb_batch, dtype=np.float32)
                frame_batch = np.reshape(np.asarray(frame_batch, dtype=np.float32),[self.batch_size*self.length,self.height,self.width,3])
                norm_frame_batch = np.asarray(norm_frame_batch, dtype=np.float32)
                pred,labels,code,_,_ = sess.run([self.pred,self.label,self.encode, self.reg_loss,self.code_loss],
                                                feed_dict={self.segnet.input_imgs:frame_batch,
                                                        self.labels:lb_batch,
                                                        self.is_training: False,
                                                         self.segnet.is_training: False,
                                                           self.segnet.keep_prob:1,
                                                        self.keep_prob:1  })
#                                      feed_dict={self.input_imgs:frame_batch,
#                                                 self.input_norm:norm_frame_batch,
#                                                    self.labels:lb_batch,
#                                                 self.is_training: False,
#                                                    self.keep_prob:1 })
                
#####################################################################################################################                
                if (n_test-2)%50 == 0:                   
                    pred_mask = sess.run([self.segnet.pred_mask],
                                          feed_dict={self.segnet.input_imgs:frame_batch,
                                                   self.segnet.is_training: True,
                                                   self.segnet.keep_prob:1  })
                    if not os.path.exists('./pred_mask/'):
                        os.makedirs('./pred_mask/')
                    pred_mask = np.asarray(pred_mask ,dtype=np.float32).reshape((self.batch_size*self.length,self.height,self.width,1)) 
                    pred_mask = pred_mask[0,:,:,:] * 255
                    pred_mask[pred_mask > 255] = 255
    #                 pred_mask = pred_mask.astype(np.uint8)
    #                 re = cv2.bitwise_and(frame_batch[0,:,:,:],frame_batch[0,:,:,:],mask = pred_mask[0,:,:,:])
                    print('############################################################################')
                    cv2.imwrite( ('./pred_mask/mask-test-'+str(epoch)+'-'+str(step) + '.jpg'),pred_mask)
#########################################################################################################################


                print('label:')
                print(labels[:2])                  
                print('pred:')
                print(pred[:2])
                pred = np.asarray(pred, dtype=np.float32)
                ppgs += pred.tolist()
                
    
                labels = np.asarray(labels, dtype=np.float32).flatten()
                ref_ppgs += labels.tolist()
                n_test += 1 
#                 code = np.asarray(code, dtype=np.float32)
#                 for i,j,k in code:
#                     ff.write(str(round(i,5))+'   '+str(round(j,5))+'     '+str(round(k,5))+'\n')
                if n_test >= thd:
                    print('cvt ppg >>>>>>>>>>>>')
                    hrs = process_data.get_hr(ppgs, self.batch_size*self.length,duration, fs=FRAME_RATE)
                    accuracy, summary = sess.run([self.hr_accuracy, summary_op], feed_dict={ 
#                         self.input_imgs:frame_batch,#frame_batch,
#                         self.input_norm:norm_frame_batch,
#                         self.labels:lb_batch,
#                         self.hrs: hrs,
#                         self.gts: gt_batch,
#                         self.is_training: False,
#                         self.keep_prob:1
                        self.segnet.input_imgs:frame_batch,
                        self.labels:lb_batch,
                        self.hrs: hrs,
                        self.gts: gt_batch,
                        self.is_training: False,
                        self.segnet.is_training: False,
                           self.segnet.keep_prob:1,
                        self.keep_prob:1 })
                    gt_accuracy += accuracy

                    ref_hrs = process_data.get_hr(ref_ppgs, self.batch_size*self.length, duration, fs=FRAME_RATE)
                    accuracy = sess.run([self.hr_accuracy], feed_dict={
#                         self.input_imgs:frame_batch,#frame_batch,
#                         self.input_norm:norm_frame_batch,
#                         self.labels:lb_batch,
#                         self.hrs: hrs,
#                         self.gts: ref_hrs,
#                         self.is_training: False,
#                         self.keep_prob:1
                        self.segnet.input_imgs:frame_batch,
                        self.labels:lb_batch,
                        self.hrs: hrs,
                        self.gts: gt_batch,
                        self.is_training: False,
                        self.segnet.is_training: False,
                        self.segnet.keep_prob:1,
                        self.keep_prob:1})[0]
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
        
        sumfile.write(str(epoch)+'-'+cond+'      '+str(round(gt_accuracy / (n_test),5))+'      '+str(round(mae,5))+'    :    '+str(round(ref_accuracy / (n_test),5))+'    '+ str(round(ref_mae,5))+'\n')
        sumfile.write('#########################################################################'+'\n')
        sumfile.close() 
        
        print('################Accuracy at epoch {0}: {1}'.format(epoch, gt_accuracy / ( n_test - thd) ))
        return step



        
        
#if __name__ == '__main__':