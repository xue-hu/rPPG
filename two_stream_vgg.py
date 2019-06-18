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
import cnn_model
import process_data

# VGG-19 parameters file
N_CLASSES = 2
FRAME_RATE = 30.0


class TwoStreamVgg(cnn_model.CnnModel):
    def construct_network(self):
        print("begin to construct two stream vgg......")
        
        with tf.name_scope('Static_Input'):
            self.input_img = tf.placeholder(dtype=tf.float32, name='input_img',
                                            shape=[self.batch_size,self.width, self.height, 3])
        with tf.name_scope('Dynamic_Input'):
            self.input_diff = tf.placeholder(dtype=tf.float32, name='input_diff',
                                             shape=[self.batch_size, self.width, self.height, 3])
            
        self.conv2d_tanh(self.input_diff,0, 32, 'conv1_1')
        self.conv2d_tanh(self.d_conv1_1, 2, 32, 'conv1_2')
        
        self.conv2d_tanh(self.input_img,0,32, 'conv1_1', stream_name='s')
        self.conv2d_tanh(self.s_conv1_1, 2, 32, 'conv1_2', stream_name='s')
        
        self.attention_layer(self.d_conv1_2, self.s_conv1_2, 'conv1_2')
        self.avgpool(self.atten_conv1_2, 'pool1')
        self.avgpool(self.s_conv1_2, 'pool1', stream_name='s')
        self.dropout_layer(self.d_pool1)

        

        self.conv2d_tanh(self.d_pool1, 5,64, 'conv2_1')
        self.conv2d_tanh(self.d_conv2_1,7, 64, 'conv2_2')
        
        self.conv2d_tanh(self.s_pool1, 5,64, 'conv2_1', stream_name='s')
        self.conv2d_tanh(self.s_conv2_1,7, 64, 'conv2_2', stream_name='s')
        
        self.attention_layer(self.d_conv2_2, self.s_conv2_2, 'conv2_2')
        self.avgpool(self.atten_conv2_2, 'pool2')
        self.dropout_layer(self.d_pool2)

        self.fully_connected_layer(self.d_pool2, 128, 'fc7')
        self.dropout_layer(self.fc7)

        self.fully_connected_layer(self.fc7, 1, 'reg_output', last_lyr=True)
        self.fully_connected_layer(self.fc7, N_CLASSES, 'class_output', last_lyr=True)
        print('param nums: '+ str(self.t))
        print("done.")  
        
    def get_data(self, video_paths, label_paths, gt_paths,clips,mode='train'):
        print("create generator....")
        batch_gen = process_data.get_batch(video_paths, label_paths, gt_paths, clips, self.batch_size,
                                           width=self.width, height=self.height, mode=mode)
        return batch_gen

    
    def inference(self):
        self.reg_output = tf.reshape(self.reg_output, shape=[self.batch_size, ])
        self.class_output = tf.reshape(self.class_output, shape=[self.batch_size, N_CLASSES]) 
        self.pred_class = tf.cast(tf.argmax(tf.nn.softmax(self.class_output),axis=1),tf.int32)
        
        
    def loss(self,model = 'reg'):
        with tf.name_scope('labels'):
            self.labels = tf.placeholder(dtype=tf.float32, name='ppg_diff', shape=[self.batch_size, ])
        print("crteate loss-Function.....")
        with tf.name_scope('loss'):
            ###########classification#####################################################################
            label_signs=tf.one_hot(tf.cast(tf.greater(self.labels,0),tf.int32),2)
            self.entropy = tf.losses.softmax_cross_entropy(onehot_labels=label_signs, logits=self.class_output)
            self.class_loss = tf.reduce_mean(self.entropy, name='class_loss')
            ###########regression##########################################################################
            self.reg_loss = tf.losses.mean_squared_error(self.reg_output, self.labels)
            ##############################################################################################
            if model == 'reg':
                self.loss = self.reg_loss 
            else:
                #self.loss = self.class_loss 
                self.loss = self.reg_loss + 0.5 * self.class_loss 
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
    
    
    def evaluation(self,model = 'reg'):
        print("create evaluation methods.....")
        with tf.name_scope('hr_accuracy'):
            self.hrs = tf.placeholder(dtype=tf.float32, name='pred_hr', shape=[self.batch_size, ])
            self.gts = tf.placeholder(dtype=tf.float32, name='ground_truth', shape=[self.batch_size, ])
            diff = tf.abs(self.hrs - self.gts)
            indicator = tf.where(diff < 3)
            total_match = tf.cast(tf.size(indicator), tf.float32)
            self.hr_accuracy = tf.truediv(total_match, tf.cast(self.batch_size, tf.float32))          
        with tf.name_scope('sign_accuracy'):
            if model == 'reg':
                right_signs = tf.greater( tf.multiply(self.reg_output, self.labels), tf.zeros_like(self.labels))
            else:
                label_signs = tf.cast(tf.greater(self.labels,0),tf.int32)
                right_signs = tf.equal( label_signs, self.pred_class )
            self.sign_accuracy = tf.reduce_mean(tf.cast(right_signs, tf.float32))        
           

    def create_summary(self):
        print("crteate summary.....")
        summary_class_loss = tf.summary.scalar('class_loss', self.class_loss) 
        summary_reg_loss = tf.summary.scalar('reg_loss', self.reg_loss) 
        summary_sign_accuracy = tf.summary.scalar('sign_accuracy', self.sign_accuracy)
        summary_hr_accuracy = tf.summary.scalar('hr_accuracy', self.hr_accuracy)
        #summary_norm = tf.summary.scalar('grads_norm', self.grads_norm)
        #summary_grad = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0]) for g in self.grads])
        summary_train = tf.summary.merge([summary_class_loss, summary_reg_loss, summary_sign_accuracy])#, summary_grad, summary_norm]) 
        summary_test = tf.summary.merge([summary_class_loss, summary_reg_loss, summary_sign_accuracy, summary_hr_accuracy]) 
        return summary_train, summary_test

    
    def train_one_epoch(self, sess, writer, saver, train_gen, summary_op, epoch, step):
        total_loss = 0
        n_batch = 0
        start_time = time.time()        
        try:
            while True:
                print("epoch " + str(epoch + 1) + "-" + str(n_batch + 1))
                (_,frames, diffs, labels, gts) = next(train_gen)
                loss,_, logits, pred_clss, labels, __, summary = sess.run([self.loss, self.class_loss,
                                                                 self.reg_output,                                                                         self.pred_class,
                                                                 self.labels, self.opt,
                                                                 summary_op],
                                                                feed_dict={self.input_img: frames,
                                                                           self.input_diff: diffs,
                                                                           self.labels: labels,
                                                                           self.keep_prob: 0.9})
                total_loss += loss
                print('label:')
                print(labels[:2])
                print('pred:')
                print(logits[:2])
                n_batch += 1
                writer.add_summary(summary, global_step=step)
                step += 1
                print('Average loss at batch {0}: {1}'.format(n_batch, total_loss / n_batch))
                if n_batch%500 == 0:                   
                    atten_map, d_conv2_2 = sess.run([self.atten_conv1_2_mask, self.d_conv2_2],feed_dict={self.input_img: frames,
                                                      self.input_diff: diffs,
                                                      self.keep_prob: 1})
                    if not os.path.exists('./atten_map/'):
                        os.makedirs('./atten_map/')
                    atten_map = np.asarray(atten_map ,dtype=np.float32).reshape((self.batch_size,self.height, self.width)) 
                    atten_map = atten_map[0,:,:] * 255
                    atten_map[atten_map > 255] = 255
                    print('############atten_map:')
                    print(atten_map[:2,:2])
                    print('############dy_conv_map2_2:')
                    print(d_conv2_2[0,:2,:2,0])
                    cv2.imwrite( ('./atten_map/'+str(epoch)+'-'+str(step) + '.jpg'),atten_map)                 
        except StopIteration:
            pass
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batch))
        print('Took:{0} seconds'.format(time.time() - start_time))
        return step,(total_loss / n_batch)


    def eval_once(self, sess, writer, test_gen,test_video_path, duration,summary_op, epoch, step,model = 'reg'):
        print("begin to evaluate.....")        
        thd = math.ceil(duration * FRAME_RATE / self.batch_size) + 1
#         for test_video_path, test_label_path, test_gt_path in zip(self.test_video_paths,self.test_label_paths, self.test_gt_paths):
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
                _,frames, diffs, labels, gts = next(test_gen)
                logits, pred_class, labels,_ = sess.run([self.reg_output, self.pred_class, self.labels, self.loss], feed_dict={self.input_img: frames,
                                                            self.input_diff: diffs,
                                                            self.labels: labels,
                                                            self.gts: gts,
                                                            self.keep_prob: 1})

                print('label:')
                print(labels[:2])                  
                if model == 'reg':
                    pred = logits.tolist()
                    print('pred:')
                    print(pred[:2])
                    ppgs += pred
                else:
                    pred = np.where(pred_class,0.1,-0.1)
                    print('pred_class:')
                    print(pred[:4]) 
                    ppgs += list(pred)

                ref_ppgs += labels.tolist()
                n_test += 1                     
                if n_test >= thd:
                    print('cvt ppg >>>>>>>>>>>>')
                    hrs = process_data.get_hr(ppgs, self.batch_size, duration, fs=FRAME_RATE)
                    accuracy, summary = sess.run([self.hr_accuracy, summary_op], feed_dict={
                        self.input_img: frames,
                        self.input_diff: diffs,
                        self.labels: labels,
                        self.hrs: hrs,
                        self.gts: gts,
                        self.keep_prob: 1})
                    gt_accuracy += accuracy

                    ref_hrs = process_data.get_hr(ref_ppgs, self.batch_size, duration, fs=FRAME_RATE)
                    accuracy = sess.run([self.hr_accuracy], feed_dict={
                        self.input_img: frames,
                        self.input_diff: diffs,
                        self.labels: labels,
                        self.hrs: hrs,
                        self.gts: ref_hrs,
                        self.keep_prob: 1})[0]
                    ref_accuracy += accuracy

                    for hr, gt, ref_hr, i, j in zip(hrs, gts, ref_hrs, pred, labels):
                        hr_li.append(hr)
                        gt_li.append(gt)
                        ref_li.append(ref_hr)
                        fileObject.write(str(round(i,2))+'      '+str(round(j,2))+'    :    '+str(int(hr))+'    '+ str(int(gt))+'    '+ str(int(ref_hr))+'\n')  

                    writer.add_summary(summary, global_step=step)
                    step += 1  
                    if n_test%50 == 0:                   
                        atten_map = sess.run([self.atten_conv1_2_mask],feed_dict={self.input_img: frames,
                                                          self.input_diff: diffs,
                                                          self.keep_prob: 1})
                        if not os.path.exists('./predict_results/'):
                            os.makedirs('./predict_results/')
                        atten_map = np.asarray(atten_map ,dtype=np.float32).reshape((self.batch_size,self.height, self.width, 1)) 
                        atten_map = atten_map[0,:,:] * 255
                        atten_map[atten_map > 255] = 255                    
                        cv2.imwrite( ('./predict_results/test-'+str(epoch)+'-'+str(step)+'-' +cond+'-'+prob_id+ '.jpg'),atten_map)    
        except StopIteration:
            pass
        mae = (np.abs(np.asarray(hr_li) - np.asarray(gt_li))).mean(axis=None)
        ref_mae = (np.abs(np.asarray(hr_li) - np.asarray(ref_li))).mean(axis=None)
        fileObject.seek(0)
        fileObject.write( 'gt_accuracy: '+str(gt_accuracy / (n_test-thd))+'\n' ) 
        fileObject.write('MAE: '+str(mae)+'\n')  
        fileObject.write('ref_accuracy: '+str(ref_accuracy / (n_test-thd))+'\n') 
        fileObject.write('ref_MAE: '+str(ref_mae)+'\n')  
        fileObject.close() 
        print('################Accuracy at epoch {0}: {1}'.format(epoch, gt_accuracy / ( n_test - thd) ))
        return step


    
if __name__ == '__main__':
    m = TwoStreamVgg(64)
    m.construct_network(112,112)