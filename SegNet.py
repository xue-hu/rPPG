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


class SegNet(cnn_model.CnnModel): 
    
    def get_data(self, video_paths, label_paths, gt_paths,clips,mode):
        print("create generator....")   
        batch_gen = process_data.get_mask_batch(video_paths, label_paths, gt_paths,
                                               clips=clips,batch_size = self.batch_size,
                                               height=self.height,width=self.width,
                                               mode=mode)       
        return batch_gen
    
    
    def construct_network(self):
        print("begin to construct SegNet......")
        
        with tf.name_scope('Atten_Input'):
            self.input_imgs = tf.placeholder(dtype=tf.float32, name='input_imgs',
                                            shape=[self.batch_size,self.height,self.width,3 ])
              
        self.conv2d_layer(self.input_imgs, 0, 64, 'conv_1',activation=tf.nn.relu,batch_norm=True, trained=False) 
        self.conv2d_layer(self.d_conv_1, 0, 64, 'down_1',stride_x=2,stride_y=2,activation=tf.nn.relu,batch_norm=True, trained=False) 

        self.conv2d_layer(self.d_down_1, 0, 128, 'conv_2',activation=tf.nn.relu,batch_norm=True, trained=False) 
        self.conv2d_layer(self.d_conv_2, 0, 128, 'down_2',stride_x=2,stride_y=2,activation=tf.nn.relu,batch_norm=True, trained=False) 

        self.conv2d_layer(self.d_down_2, 0, 256, 'conv_3',activation=tf.nn.relu,batch_norm=True, trained=False) 
        self.conv2d_layer(self.d_conv_3, 0, 256, 'down_3',stride_x=2,stride_y=2,activation=tf.nn.relu,batch_norm=True, trained=False)

        self.conv2d_layer(self.d_down_3, 0, 512, 'conv_4',activation=tf.nn.relu,batch_norm=True, trained=False) 
        self.conv2d_layer(self.d_conv_4, 0, 512, 'down_4',stride_x=2,stride_y=2,activation=tf.nn.relu,batch_norm=True, trained=False)

        self.trans_conv2d_layer(self.d_down_4, 0, 512, 'up_4',activation=tf.nn.relu,batch_norm=True, trained=False) 
        self.conv2d_layer(self.d_up_4, 0, 512, 'tconv_4',activation=tf.nn.relu,batch_norm=True, trained=False) 

        self.trans_conv2d_layer(self.d_tconv_4, 0, 256, 'up_3',activation=tf.nn.relu,batch_norm=True, trained=False) 
        self.conv2d_layer(self.d_up_3, 0, 256, 'tconv_3',activation=tf.nn.relu,batch_norm=True, trained=False) 

        self.trans_conv2d_layer(self.d_tconv_3, 0, 128, 'up_2',activation=tf.nn.relu,batch_norm=True, trained=False) 
        self.conv2d_layer(self.d_up_2, 0, 138, 'tconv_2',activation=tf.nn.relu,batch_norm=True, trained=False)

        self.trans_conv2d_layer(self.d_tconv_2, 0, 64, 'up_1',activation=tf.nn.relu,batch_norm=True, trained=False) 
        self.conv2d_layer(self.d_up_1, 0, 64, 'tconv_1',activation=tf.nn.relu,batch_norm=True, trained=False)

        self.conv2d_layer(self.d_tconv_1, 0, 2, 'pred',activation=tf.nn.relu,batch_norm=True, trained=False)
       
    def inference(self):        
        with tf.variable_scope('predictions'):
            self.pred = tf.reshape(self.d_pred, shape=[self.batch_size,self.height,self.width,2])
            self.pred_mask = tf.cast(tf.argmax(self.pred,3),tf.float32)
                            
    def loss(self,model=None):
        print("crteate loss-Function.....")
        with tf.name_scope('loss'):
            self.mask = tf.placeholder(dtype=tf.int32, name='mask', 
                                       shape=[self.batch_size,self.height,self.width,1])
            mask = tf.reshape(tf.one_hot(self.mask, 2, axis=3), [-1, 2])
            pred = tf.reshape(self.pred, shape=[-1, 2])
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=mask))
            mask = tf.reshape(tf.one_hot(self.mask, 2, axis=3), [self.batch_size,-1, 2])
            pred = tf.reshape(self.pred, shape=[self.batch_size,-1, 2])
            self.iou, self.update_op = tf.metrics.mean_iou(tf.argmax(pred,2), tf.argmax(mask,2), 2, name='iou')

            
            
    def optimizer(self,gstep):
        print("crteate optimizer.....")
        with tf.control_dependencies( tf.get_collection( tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
            self.opt = optimizer.minimize(self.loss,global_step=gstep)

            
    
    
    def evaluation(self, model=None):
        print("create evaluation methods.....")
                   
  
           

    def create_summary(self):
        print("crteate summary.....")
        summary_loss = tf.summary.scalar('reg_loss', self.loss) 
        summary_iou = tf.summary.scalar('iou', self.iou) 
        summary_train = tf.summary.merge([summary_loss])#, summary_grad, summary_norm]) 
        summary_test = tf.summary.merge([summary_iou,summary_loss]) 
        return summary_train, summary_test
    
    def build_graph(self):
        self.construct_network()
        self.inference()
        self.loss()
        self.evaluation()

 

    def train_one_epoch(self, sess, writer, saver, train_data,summary_op, epoch, step):
        total_loss = 0
        n_batch = 1
        start_time = time.time()   
        for (frame_batch,mask_batch,_,_) in train_data:
            print("epoch " + str(epoch + 1) + "-" + str(n_batch))               
            frame_batch = np.asarray(frame_batch, dtype=np.float32)
            mask_batch = np.expand_dims(np.asarray(mask_batch, dtype=np.float32),-1)
            pred,loss,_,_,summary = sess.run([self.pred,
                                 self.loss,
                                 self.opt,
                                 self.update_op,
                                 summary_op], 
                                 feed_dict={self.input_imgs:frame_batch,
                                            self.mask:mask_batch,
                                            self.is_training: True,
                                               self.keep_prob:1  })
            total_loss += loss
            writer.add_summary(summary, global_step=step)
            step += 1
            print('Average loss at batch {0}: {1}'.format(n_batch, total_loss / n_batch))
            n_batch += 1      
####################################################################################################################
            if (n_batch-2)%50 == 0:                   
                pred_mask = sess.run([self.pred_mask],
                                      feed_dict={self.input_imgs:frame_batch,
                                                 self.mask:mask_batch,
                                               self.is_training: True,
                                               self.keep_prob:1  })
                if not os.path.exists('./pred_mask/'):
                    os.makedirs('./pred_mask/')
                pred_mask = np.asarray(pred_mask ,dtype=np.float32).reshape((self.batch_size,self.height,self.width,1)) 
                pred_mask = pred_mask[0,:,:,:] * 255
                pred_mask[pred_mask > 255] = 255
#                 pred_mask = pred_mask.astype(np.uint8)
#                 re = cv2.bitwise_and(frame_batch[0,:,:,:],frame_batch[0,:,:,:],mask = pred_mask[0,:,:,:])
                print('############################################################################')
                cv2.imwrite( ('./pred_mask/mask-'+str(epoch)+'-'+str(step) + '.jpg'),pred_mask)
#########################################################################################################################
        
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batch))
        print('Took:{0} seconds'.format(time.time() - start_time))
        return step,(total_loss / n_batch)

        
        
    def eval_once(self, sess, writer, test_gen,test_video_path, duration, summary_op, epoch, step):
        print("begin to evaluate.....")        
        path = test_video_path.split('/')
        prob_id = path[4]
        cond = path[5].split('_')[0]
        if not os.path.exists('./predict_results/'):
            os.makedirs('./predict_results/')
        n_test = 0
        try:
            while True:
                frame_batch,mask_batch,_,_ = next(test_gen)               
                frame_batch = np.asarray(frame_batch, dtype=np.float32)
                mask_batch = np.expand_dims(np.asarray(mask_batch, dtype=np.float32),-1)
                pred,pred_mask= sess.run([self.pred,self.pred_mask], 
                                     feed_dict={self.input_imgs:frame_batch,
                                                self.mask:mask_batch,
                                                self.is_training: False,
                                                   self.keep_prob:1 })
                
#####################################################################################################################                
                if (n_test-2)%100 == 0:                                       
                    if not os.path.exists('./pred_mask/'):
                        os.makedirs('./pred_mask/')
                    pred_mask = np.asarray(pred_mask ,dtype=np.float32).reshape((self.batch_size,self.height,self.width, 1)) 
                    pred_mask = pred_mask[0,:,:,:] * 255
                    pred_mask[pred_mask > 255] = 255
                    print('############################################################################')
                    cv2.imwrite( ('./pred_mask/mask-'+str(epoch)+'-'+str(step) + '.jpg'),pred_mask)
######################################################################################################################### 
                writer.add_summary(summary, global_step=step)
                step += 1 
                n_test += 1
        except StopIteration:
            pass        
        return step



        
        
#if __name__ == '__main__':