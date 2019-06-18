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
import process_data

# VGG-19 parameters file
N_CLASSES = 2
FRAME_RATE = 30.0


class FaceVgg(cnn_model.CnnModel): 
    
    def get_data(self, video_paths, label_paths, gt_paths,clips,mode):
        print("create generator....")   
        batch_gen = process_data.get_batch(video_paths, label_paths, gt_paths,clips=clips, batch_size=self.batch_size,mode=mode,width=self.width, height=self.height)    
        return batch_gen
    
    
    def construct_network(self):
        print("begin to construct face-vgg......")
        
        with tf.name_scope('Input'):
            self.input_imgs = tf.placeholder(dtype=tf.float32, name='input_imgs',
                                            shape=[None,self.height, self.width, 3])

####################### original pretrained Face-VGG #####################################################################          
#         self.conv2d_relu(self.s_pool3, 17, 'conv4_1',trainable=True)
#         self.conv2d_relu(self.d_conv4_1, 19, 'conv4_2',trainable=True)
#         self.conv2d_relu(self.d_conv4_2, 21, 'conv4_3',trainable=True)
#         self.avgpool(self.d_conv4_3, 'pool4', stream_name='s')
#         self.dropout_layer(self.s_pool4)
        
#         self.conv2d_relu(self.s_pool4, 24, 'conv5_1',trainable=True)
#         self.conv2d_relu(self.d_conv5_1, 26, 'conv5_2',trainable=True)
#         self.conv2d_relu(self.d_conv5_2, 28, 'conv5_3',trainable=True)
        #self.avgpool(self.d_conv5_3, 'pool5', stream_name='s')
        #self.dropout_layer(self.s_pool5)
        
        #self.conv2d_relu(self.d_conv5_3, 31, 'fc6')
        #self.conv2d_relu(self.fc6, 33, 'fc7')
        
################## 2-stream LTRCN ##########################################################################

#         self.conv2d_layer(self.input_imgs, 0, 32, 'conv1_1',trained=False)
#         self.conv2d_layer(self.d_conv1_1, 2, 32, 'conv1_2',trained=False)
#         self.conv2d_layer(self.input_imgs, 0, 32, 'conv1_1',stream_name='s')
#         self.conv2d_layer(self.s_conv1_1, 2, 32, 'conv1_2',stream_name='s')
#         self.attention_layer(self.d_conv1_2, self.s_conv1_2, 'd_conv1_2', 's_conv1_2')
#         self.avgpool(self.s_conv1_2, 'pool1', stream_name='s')
#         self.avgpool(self.atten_conv1_2, 'pool1')
#         self.dropout_layer(self.d_pool1)
        
#         self.conv2d_layer(self.d_pool1,5, 64, 'conv2_1',trained=False)
#         self.conv2d_layer(self.d_conv2_1,7, 64, 'conv2_2',trained=False)
#         self.conv2d_layer(self.s_pool1,5, 64, 'conv2_1',stream_name='s')
#         self.conv2d_layer(self.s_conv2_1,7, 64, 'conv2_2',stream_name='s')
#         self.attention_layer(self.d_conv2_2, self.s_conv2_2, 'd_conv2_2', 's_conv2_2')
#         self.avgpool(self.atten_conv2_2, 'pool2')
#         self.dropout_layer(self.d_pool2)
        
# #         self.conv2d_relu(self.d_pool2, 10, 'conv3_1',trainable=True)
# #         self.conv2d_relu(self.d_conv3_1, 12, 'conv3_2',trainable=True)
# #         self.avgpool(self.d_conv3_2, 'pool3')
# #         self.dropout_layer(self.d_pool3)        
        
#         self.fully_connected_layer(self.d_pool2, 128, 'fc6',last_lyr=False)
####################1-stream LTRCN + maseked Input ######################################################################
    
        self.conv2d_layer(self.input_imgs, 0, 16, 'conv1_1',trained=False,batch_norm=True)
        self.conv2d_layer(self.d_conv1_1, 2, 16, 'conv1_2',trained=False)
        self.avgpool(self.d_conv1_2, 'pool1')
        self.dropout_layer(self.d_pool1)
        
        self.conv2d_layer(self.d_pool1,5, 32, 'conv2_1',trained=False,batch_norm=True)
        self.conv2d_layer(self.d_conv2_1,7, 32, 'conv2_2',trained=False)        
        self.avgpool(self.d_conv2_2, 'pool2')
        self.dropout_layer(self.d_pool2)
        
#         self.conv2d_layer(self.d_pool2, 10,32, 'conv3_1',trained=False)
#         self.conv2d_layer(self.d_conv3_1, 12,64, 'conv3_2',trained=False)
#         self.avgpool(self.d_conv3_1, 'pool3')
#         self.dropout_layer(self.d_pool3)        
        
        self.fully_connected_layer(self.d_pool2, 128, 'fc6',last_lyr=False)

    
       
    def inference(self):
        self.fully_connected_layer(self.fc6, 1, 'reg_output',last_lyr=True)
        self.fully_connected_layer(self.fc6, 2, 'class_output',last_lyr=True)
        with tf.name_scope('predictions'):
            self.pred = tf.reshape(self.reg_output, shape=[self.batch_size,])
            self.logits = tf.reshape(self.class_output, shape=[self.batch_size, 2])
            self.pred_class = tf.cast(tf.argmax(tf.nn.softmax(self.logits),axis=1),tf.int32)

                            
    def loss(self,model=None):
        with tf.name_scope('labels'):
            self.labels = tf.placeholder(dtype=tf.float32, name='ppg_diff', shape=[self.batch_size, ])
        print("crteate loss-Function.....")
        with tf.name_scope('loss'):
            label_signs=tf.one_hot(tf.cast(tf.greater(self.labels,0),tf.int32),2)
            self.entropy = tf.losses.softmax_cross_entropy(onehot_labels=label_signs, logits=self.logits)
            self.class_loss = tf.reduce_mean(self.entropy, name='class_loss')
            self.reg_loss = tf.losses.mean_squared_error(self.pred, self.labels)
            w_regulation = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = self.reg_loss + tf.reduce_sum(w_regulation) #+ self.class_loss 
            
    def optimizer(self,gstep):
        print("crteate optimizer.....")
        with tf.control_dependencies( tf.get_collection( tf.GraphKeys.UPDATE_OPS)):
#         lr = tf.train.exponential_decay(self.lr, gstep,5000, 0.96,staircase=True)
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
#         grads, variable_names = zip(*optimizer.compute_gradients(self.loss))
#         clip_grads,_ = tf.clip_by_global_norm(grads, 20)
#         self.grads = [(g, v) for g,v in zip(clip_grads, variable_names)]
#         self.grads_norm = tf.global_norm([g[0] for g in self.grads])
#         self.opt = optimizer.apply_gradients(self.grads, global_step=gstep)
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
        summary_loss = tf.summary.scalar('reg_loss', self.reg_loss) 
        summary_class_loss = tf.summary.scalar('class_loss', self.class_loss)
        summary_hr_accuracy = tf.summary.scalar('hr_accuracy', self.hr_accuracy)
        summary_hr_mae = tf.summary.scalar('hr_mae', self.hr_mae)
#         summary_norm = tf.summary.scalar('grads_norm', self.grads_norm)
#         summary_grad = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in self.grads])
        summary_train = tf.summary.merge([summary_class_loss,summary_loss])#, summary_grad, summary_norm]) 
        summary_test = tf.summary.merge([summary_class_loss,summary_loss, summary_hr_accuracy, summary_hr_mae]) 
        return summary_train, summary_test
    
    
    def train_one_epoch(self, sess, writer, saver, train_data,summary_op, epoch, step):
        total_loss = 0
        n_batch = 1
        start_time = time.time()        
        for (frame_batch,lb_batch, gt_batch) in train_data:
            print("epoch " + str(epoch + 1) + "-" + str(n_batch))               
            lb_batch = np.asarray(lb_batch, dtype=np.float32)#[:,-1]
            gt_batch = np.asarray(gt_batch, dtype=np.float32)#[:,-1]
            frame_batch = np.asarray(frame_batch, dtype=np.float32)
            
            pred,loss,_,labels,_,summary = sess.run([self.pred,
                                 self.reg_loss,
                                 self.class_loss,
                                 self.labels,
                                 self.opt,
                                 summary_op], 
                                 feed_dict={self.input_imgs:frame_batch,
                                               self.labels:lb_batch,
                                            self.is_training: True,
                                               self.keep_prob:0.8  })
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
#                 if (n_batch-2)%50 == 0:                   
#                     atten_map = sess.run([self.atten_conv1_2_mask],feed_dict={self.input_imgs:pre_fra_batch,
#                                                    self.keep_prob:1  })
#                     if not os.path.exists('./atten_map/'):
#                         os.makedirs('./atten_map/')
#                     atten_map = np.asarray(atten_map ,dtype=np.float32).reshape((self.batch_size,self.height, self.width)) 
#                     atten_map = atten_map[0,:,:] * 255
#                     atten_map[atten_map > 255] = 255
#                     print('############atten_map:')
#                     print(atten_map[:2,:2])
#                     cv2.imwrite( ('./atten_map/att-'+str(epoch)+'-'+str(step) + '.jpg'),atten_map)             
##########################################################################################################################
        
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
        sumfile = open('./predict_results/'+prob_id+'-summary.txt', 'a')
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
                frame_batch,_,lb_batch,_, gt_batch = next(test_gen)               
                gt_batch = np.asarray(gt_batch, dtype=np.float32)#[:,-1]
                lb_batch = np.asarray(lb_batch, dtype=np.float32)#[:,-1]
                frame_batch = np.asarray(frame_batch, dtype=np.float32)
                pred,labels,_,_ = sess.run([self.pred,self.labels, self.reg_loss,self.class_loss], 
                                     feed_dict={self.input_imgs:frame_batch,
                                                   self.labels:lb_batch,
                                                self.is_training: False,
                                                   self.keep_prob:1 })
                
#####################################################################################################################                
#                 if n_test%2000 == 0:                   
#                     atten_map = sess.run([self.atten_conv1_2_mask],feed_dict={self.input_1:pre_fra_batch,
#                                             self.input_2:next_fra_batch,
#                                             self.input_3:diff_batch,
#                                                    self.keep_prob:1  })
#                     if not os.path.exists('./predict_results/'):
#                         os.makedirs('./predict_results/')
#                     atten_map = np.asarray(atten_map ,dtype=np.float32).reshape((self.batch_size,self.height, self.width)) 
#                     atten_map = atten_map[0,:,:] * 255
#                     atten_map[atten_map > 255] = 255
#                     print('############atten_map:')
#                     print(atten_map[:2,:2])
#                     cv2.imwrite( ('./predict_results/test-'+str(epoch)+'-'+str(step)+'-' +cond+'-'+prob_id+ '.jpg'),atten_map) 
                
#########################################################################################################################

                print('label:')
                print(labels[:2])                  
                print('pred:')
                print(pred[:2])
                ppgs += pred.tolist()
                
    
                ref_ppgs += labels.tolist()
                n_test += 1                     
                if n_test >= thd:
                    print('cvt ppg >>>>>>>>>>>>')
                    hrs = process_data.get_hr(ppgs, self.batch_size,duration, fs=FRAME_RATE)
#                     hrs = np.asarray(ppgs, dtype=np.float32) * 12.279 + 71.58
                    accuracy, summary = sess.run([self.hr_accuracy, summary_op], feed_dict={ 
                        self.input_imgs:frame_batch,#frame_batch,
                        self.labels:lb_batch,
                        self.hrs: hrs,
                        self.gts: gt_batch,
                        self.is_training: False,
                        self.keep_prob:1 })
                    gt_accuracy += accuracy

                    ref_hrs = process_data.get_hr(ref_ppgs, self.batch_size, duration, fs=FRAME_RATE)
#                     ref_hrs = np.asarray(ref_ppgs, dtype=np.float32) * 12.279 + 71.58
                    accuracy = sess.run([self.hr_accuracy], feed_dict={
                        self.input_imgs:frame_batch,#frame_batch,
                        self.labels:lb_batch,
                        self.hrs: hrs,
                        self.gts: ref_hrs,
                        self.is_training: False,
                        self.keep_prob:1 })[0]
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
   