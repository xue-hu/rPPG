#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import time
import tensorflow as tf
import cv2
import loading_model
import process_data
import utils
import math
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ECG_SAMPLE_RATE = 16.0
PLE_SAMPLE_RATE = 256.0
FRAME_RATE = 30.0
N_CLASSES = 2


class VideoAnalysis(object):
    def __init__(self, train_video_paths, train_label_paths, train_gt_paths,
                 test_video_paths, test_label_paths, test_gt_paths,
                 img_width, img_height):
        self.train_video_paths = train_video_paths
        self.train_label_paths = train_label_paths
        self.train_gt_paths = train_gt_paths
        self.test_video_paths = test_video_paths
        self.test_label_paths = test_label_paths
        self.test_gt_paths = test_gt_paths
        self.width = img_width
        self.height = img_height
        self.duration = 12
        self.lr = 0.1
        self.sign_loss_weight = 1.0
        self.batch_size = 64
        self.gstep = tf.Variable(0, trainable=False, name='global_step')
        self.skip_step = 50000

    def get_data(self, video_paths, label_paths, gt_paths, clips, batch_size, mode='train'):
        print("create generator....")
        batch_gen = process_data.get_batch(video_paths, label_paths, gt_paths, clips, batch_size,
                                           width=self.width, height=self.height, mode=mode)
        return batch_gen

    def loading_model(self):
        with tf.name_scope('Static_Input'):
            self.input_img = tf.placeholder(dtype=tf.float32, name='input_img',
                                            shape=[self.batch_size, self.height, self.width, 3])
        with tf.name_scope('Dynamic_Input'):
            self.input_diff = tf.placeholder(dtype=tf.float32, name='input_diff',
                                             shape=[self.batch_size, self.height, self.width, 3])
        with tf.name_scope('labels'):
            self.labels = tf.placeholder(dtype=tf.float32, name='ppg_diff', shape=[self.batch_size, ])
        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob', shape=[])
        self.model = loading_model.NnModel(self.input_img, self.input_diff, self.keep_prob)
        self.model.two_stream_vgg_load()


    def inference(self):
        self.reg_output = tf.reshape(self.model.reg_output, shape=[self.batch_size, ])
        self.class_output = tf.reshape(self.model.class_output, shape=[self.batch_size, N_CLASSES])        

    def loss(self):
        print("crteate loss-Function.....")
        with tf.name_scope('loss'):
            ###########classification#####################################################################
            label_signs=tf.one_hot(tf.cast(tf.greater(self.labels,0),tf.int32),2)
            self.entropy = tf.losses.softmax_cross_entropy(onehot_labels=label_signs, logits=self.class_output)
            self.class_loss = tf.reduce_mean(self.entropy, name='class_loss')
            ###########regression##########################################################################
            self.reg_loss = tf.losses.mean_squared_error(self.reg_output, self.labels)
            ##############################################################################################
            self.loss = self.reg_loss + 1.0 * self.class_loss          
            #self.loss = tf.losses.huber_loss(self.logits, self.labels)# + self.sign_loss_weight*self.sign_loss            
  

    def optimizer(self):
        print("crteate optimizer.....")
        optimizer = tf.train.AdadeltaOptimizer(self.lr)
#         grads, variable_names = zip(*optimizer.compute_gradients(self.loss))
#         clip_grads,_ = tf.clip_by_global_norm(grads, 500)
#         self.grads = [(g, v) for g,v in zip(clip_grads, variable_names)]
#         self.grads_norm = tf.global_norm([g[0] for g in self.grads])
#         self.opt = optimizer.apply_gradients(self.grads, global_step=self.gstep)
        self.opt = optimizer.minimize(self.loss, global_step=self.gstep)
        

    def evaluation(self):
        print("create evaluation methods.....")
        with tf.name_scope('hr_accuracy'):
            self.hrs = tf.placeholder(dtype=tf.float32, name='pred_hr', shape=[self.batch_size, ])
            self.gts = tf.placeholder(dtype=tf.float32, name='ground_truth', shape=[self.batch_size, ])
            diff = tf.abs(self.hrs - self.gts)
            indicator = tf.where(diff < 3)
            total_match = tf.cast(tf.size(indicator), tf.float32)
            self.hr_accuracy = tf.truediv(total_match, tf.cast(self.batch_size, tf.float32))          
        with tf.name_scope('sign_accuracy'):
            false_signs = tf.greater( tf.multiply(self.reg_output, self.labels), tf.zeros_like(self.labels))
            self.sign_accuracy = tf.reduce_mean(tf.cast(false_signs, tf.float32))        
           

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


    def build_graph(self):
        self.loading_model()
        self.inference()
        self.loss()
        self.evaluation()
        self.optimizer()


    def train_one_epoch(self, sess, writer, saver, summary_op, epoch, step):
        total_loss = 0
        n_batch = 0
        start_time = time.time()
        train_gen = self.get_data(self.train_video_paths, self.train_label_paths, self.train_gt_paths, np.arange(2, 601), batch_size=self.batch_size)
        try:
            while True:
                print("epoch " + str(epoch + 1) + "-" + str(n_batch + 1))
                (frames, diffs, labels, gts) = next(train_gen)
                loss,_, logits, labels, __, summary = sess.run([self.loss, self.class_loss,
                                                                 self.reg_output,
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
                    atten_map, d_conv2_2 = sess.run([self.model.atten_conv1_2_mask, self.model.d_conv2_2],feed_dict={self.input_img: frames,
                                                      self.input_diff: diffs,
                                                      self.keep_prob: 1})
                    if not os.path.exists('./n_processed_video/atten_map/'):
                        os.makedirs('./n_processed_video/atten_map/')
                    atten_map = np.asarray(atten_map ,dtype=np.float32).reshape((self.batch_size,self.height, self.width)) 
                    atten_map = atten_map[0,:,:] * 255
                    atten_map[atten_map > 255] = 255
                    print('############atten_map:')
                    print(atten_map[:2,:2])
                    print('############dy_conv_map2_2:')
                    print(d_conv2_2[0,:2,:2,0])
                    cv2.imwrite( ('./n_processed_video/atten_map/'+str(epoch)+'-'+str(step) + '.jpg'),atten_map)                 
        except StopIteration:
            pass
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batch))
        print('Took:{0} seconds'.format(time.time() - start_time))
        return step,(total_loss / n_batch)


    def eval_once(self, sess, writer, summary_op, epoch, step):
        print("begin to evaluate.....")        
        thd = math.ceil(self.duration * FRAME_RATE / self.batch_size) + 1
        for test_video_path, test_label_path, test_gt_path in zip(self.test_video_paths,self.test_label_paths, self.test_gt_paths):
            path = test_video_path.split('/')
            prob_id = path[4]
            cond = path[5].split('_')[0]
            if not os.path.exists('./predict_results/'):
                os.makedirs('./predict_results/')
            fileObject = open('./predict_results/'+cond+'-'+prob_id+'-ppg('+str(epoch)+').txt', 'w')               
 

            test_gen = self.get_data([test_video_path], [test_label_path], [test_gt_path], [1, 2], batch_size=self.batch_size, mode='test')
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
                    frames, diffs, labels, gts = next(test_gen)
                    logits, labels = sess.run([self.reg_output, self.labels], feed_dict={self.input_img: frames,
                                                                self.input_diff: diffs,
                                                                self.labels: labels,
                                                                self.gts: gts,
                                                                self.keep_prob: 1})

                    print('label:')
                    print(labels[:2])
                    pred = logits.tolist()
                    print('pred:')
                    print(pred[:2])                    
                    ppgs += pred
                    ref_ppgs += labels.tolist()
                    n_test += 1                     
                    if n_test >= thd:
                        print('cvt ppg >>>>>>>>>>>>')
                        hrs = process_data.get_hr(ppgs, self.batch_size, self.duration, fs=FRAME_RATE)
                        accuracy, summary = sess.run([self.hr_accuracy, summary_op], feed_dict={
                            self.input_img: frames,
                            self.input_diff: diffs,
                            self.labels: labels,
                            self.hrs: hrs,
                            self.gts: gts,
                            self.keep_prob: 1})
                        gt_accuracy += accuracy
                        
                        ref_hrs = process_data.get_hr(ref_ppgs, self.batch_size, self.duration, fs=FRAME_RATE)
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
                            fileObject.write(str(round(i,2))+'      '+str(round(j,2))+'    :    '+str(int(hr))+'    '+ str(int(gt))+'    '+ str(int(ref_hr)))  
                            fileObject.write('\n') 
                        
                        writer.add_summary(summary, global_step=step)
                        step += 1  
                    if n_test%50 == 0:                   
                        atten_map = sess.run([self.model.atten_conv1_2_mask],feed_dict={self.input_img: frames,
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
            fileObject.write('gt_accuracy: '+str(gt_accuracy / (n_test-thd))) 
            fileObject.write('\n')
            fileObject.write('MAE: '+str(mae))  
            fileObject.write('\n')
            fileObject.write('ref_accuracy: '+str(ref_accuracy / (n_test-thd))) 
            fileObject.write('\n')
            fileObject.write('ref_MAE: '+str(ref_mae))  
            fileObject.write('\n')
            fileObject.close() 
            print('################Accuracy at epoch {0}: {1}'.format(epoch, gt_accuracy / ( n_test - thd) ))


    def train(self, n_epoch):
        print("begin to train.....")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("initialization completed.")
            writer = tf.summary.FileWriter('./graphs/', sess.graph)
            print("computational graph saved.")
            saver = tf.train.Saver()
            summary_train, summary_test = self.create_summary()
            step = self.gstep.eval()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoint_dict/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('loading stored params...........')
            pre_loss = 100
            for epoch in range(n_epoch): 
                step, loss = self.train_one_epoch(sess, writer, saver, summary_train, epoch, step)                
                saver.save(sess, './checkpoint_dict/', global_step=self.gstep)
                if pre_loss <= loss:
                    self.lr = self.lr / 2.0
                pre_loss = loss
                self.eval_once(sess, writer, summary_test, epoch, step)                            
            writer.close()


if __name__ == '__main__':
    ############using remote dataset######################################################
    tr_vd_paths = []
    tr_lb_paths = []
    tr_gt_paths = []
    te_vd_paths = []
    te_lb_paths = []
    te_gt_paths = []
    t_id = 6
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
    for cond in ['lighting']:
        if cond == 'lighting':
            n = [1]
        for i in n:
            tr_vd_path, tr_lb_path = utils.create_file_paths(np.delete(np.arange(1, 27), [9,20,22,24,    t_id-1  ]), cond=cond, cond_typ=i)
            _, tr_gt_path = utils.create_file_paths(np.delete(np.arange(1, 27), [9,20,22,24,  t_id-1   ]), cond=cond, cond_typ=i, sensor_sgn=0)
            te_vd_path, te_lb_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i)
            _, te_gt_path = utils.create_file_paths([t_id], cond=cond, cond_typ=i, sensor_sgn=0)
            tr_vd_paths += tr_vd_path
            tr_lb_paths += tr_lb_path
            tr_gt_paths += tr_gt_path
            te_vd_paths += te_vd_path
            te_lb_paths += te_lb_path
            te_gt_paths += te_gt_path
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
    


    model = VideoAnalysis(tr_vd_paths, tr_lb_paths, tr_gt_paths, te_vd_paths, te_lb_paths, te_gt_paths, img_height=64, img_width=64 )
    model.build_graph()
    model.train(20)
