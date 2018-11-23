#!/usr/bin/python3.5
__author__ = 'Iris'

import os
import time
import tensorflow as tf
import cv2
import loading_model
import process_data
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_LABELS = 3862
TRAIN_VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']
TRAIN_LABEL_PATHS = ['D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin']
TEST_VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']
TEST_LABEL_PATHS = ['D:/PycharmsProject/yutube8M/data/synced_Logitech HD Pro Webcam C920/5_Pleth.bin']


class VideoAnalysis(object):
    def __init__(self, train_video_paths, train_label_paths,
                 test_video_paths, test_label_paths,
                 img_width=256, img_height=256):
        self.train_video_paths = train_video_paths
        self.train_label_paths = train_label_paths
        self.test_video_paths = test_video_paths
        self.test_label_paths = test_label_paths
        self.width = img_width
        self.height = img_height
        self.lr = 4.0
        self.batch_size = 32
        self.gstep = tf.Variable(0, trainable=False, name='global_step')
        self.skip_step = 3000

    def get_data(self, video_paths, label_paths, clips):
        print("create generator....")
        batch_gen = process_data.get_batch(video_paths, label_paths, clips, self.batch_size, width=self.width, height=self.height)
        return batch_gen

    # def create_training_input(self, train_video_path, train_label_path):
    #     print("create new training-set generator.....")
    #     self.train_gen = self.get_data(train_video_path, train_label_path)
    #
    # def create_testing_input(self, test_video_path, test_label_path):
    #     print("create new testing-set generator.....")
    #     self.test_gen = self.get_data(test_video_path, test_label_path)

    def loading_model(self):
        with tf.name_scope('Static_Input'):
            self.input_img = tf.placeholder(dtype=tf.float32, name='input_img',
                                        shape=[self.batch_size, self.width, self.height, 3])
        with tf.name_scope('Dynamic_Input'):
            self.input_diff = tf.placeholder(dtype=tf.float32, name='input_diff',
                                         shape=[self.batch_size, self.width, self.height, 3])
        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_prob', shape=[])
        self.model = loading_model.NnModel(self.input_img, self.input_diff, self.keep_prob)
        #self.model.vgg_load()
        self.model.two_stream_vgg_load()

    def inference(self):
        self.pred = tf.reshape(self.model.output, [self.batch_size, ])

    def loss(self):
        print("crteate loss-Function.....")
        with tf.name_scope('loss'):
            self.labels = tf.placeholder(dtype=tf.float32, name='input', shape=[self.batch_size, ])
            self.loss = tf.losses.mean_squared_error(labels=self.labels, predictions=self.pred)

    def evaluation(self):
        print("create evaluation methods.....")
        with tf.name_scope('accuracy'):
            diff = tf.abs(self.pred - self.labels)
            indicator = tf.where(diff < 10)
            self.accuracy = tf.truediv(tf.size(indicator), self.batch_size)
            #self.accuracy = tf.losses.mean_squared_error(labels=self.labels, predictions=self.pred)/ self.batch_size

    def optimizer(self):
        print("crteate optimizer.....")
        self.opt = tf.train.AdadeltaOptimizer(self.lr).minimize(self.loss, global_step=self.gstep)

    def create_summary(self):
        print("crteate summary.....")
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        summary_op = tf.summary.merge_all()
        return summary_op

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
        train_gen = self.get_data(self.train_video_paths, self.train_label_paths, [1, 2, 3, 4, 5])
        try:
            while True:
                print("epoch " + str(epoch + 1) + "-" + str(n_batch + 1))
                (frames, diffs, labels) = next(train_gen)
                ##########testing the generator##############
                # for frame, diff, label in zip(frames, diffs, labels):
                #     cv2.imshow('face', frame)
                #     cv2.imshow('diff', diff)
                #     print(label)
                #     cv2.waitKey(0)
                ############################################
                loss, _, summary = sess.run([self.loss, self.opt, summary_op],
                                            feed_dict={self.input_img: frames,
                                                       self.input_diff: diffs,
                                                       self.labels: labels,
                                                       self.keep_prob: 0.9})
                total_loss += loss
                n_batch += 1
                writer.add_summary(summary, global_step=step)
                step += 1
                print('Average loss at batch {0}: {1}'.format(n_batch, total_loss / n_batch))
                if step % self.skip_step == 0:
                    saver.save(sess, './checkpoint_dict/', global_step=self.gstep)
        except StopIteration:
            pass
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batch))
        print('Took:{0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, writer, summary_op, epoch, step):
        print("begin to evaluate.....")
        total_accuracy = 0
        n_test = 0
        test_gen = self.get_data(self.test_video_paths, self.test_label_paths, [1, 3])
        try:
            while True:
                frames, diffs, labels = next(test_gen)
                accuracy, summary = sess.run([self.accuracy, summary_op],
                                             feed_dict={self.input_img: frames,
                                                        self.input_diff: diffs,
                                                        self.labels: labels,
                                                        self.keep_prob: 1})
                total_accuracy += accuracy
                n_test += 1
                writer.add_summary(summary, global_step=step)
        except StopIteration:
            pass
        print('Accuracy at epoch {0}: {1}'.format(epoch, total_accuracy / n_test))

    def train(self, n_epoch):
        print("begin to train.....")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("initialization completed.")
            writer = tf.summary.FileWriter('./graphs/', sess.graph)
            print("computational graph saved.")
            saver = tf.train.Saver()
            summary_op = self.create_summary()
            step = self.gstep.eval()
            # if tf.train.checkpoint_exists('./checkpoint'):
            #     saver.restore(sess, './checkpoint')
            for epoch in range(n_epoch):
                step = self.train_one_epoch(sess, writer, saver, summary_op, epoch, step)
                self.eval_once(sess, writer, summary_op, epoch, step)
            writer.close()


if __name__ == '__main__':
    ############using remote dataset######################################################
#    tr_vd_paths = []
#    tr_lb_paths = []
#    te_vd_paths = []
#    te_lb_paths = []
#    for cond in ['lighting','movement']:
#        if cond == 'lighting':
#            n = 6
#        else:
#            n = 4
#        for i in range(1):
#            tr_vd_path, tr_lb_path = utils.create_file_paths([2], cond=cond, cond_typ=i)
#            te_vd_path, te_lb_path = utils.create_file_paths([4], cond=cond, cond_typ=i)
#            tr_vd_paths += tr_vd_path
#            tr_lb_paths += tr_lb_path
#            te_vd_paths += te_vd_path
#            te_lb_paths += te_lb_path
    p = range(12, 15)
    s_p = [2, 3, 4, 6, 7, 9, 10]
    s_p += p
    tr_vd_paths, tr_lb_paths = utils.create_file_paths(s_p)
    te_vd_paths, te_lb_paths = utils.create_file_paths([5])
    model = VideoAnalysis(tr_vd_paths, tr_lb_paths, te_vd_paths, te_lb_paths)
    ######################################################################################
    #model = VideoAnalysis(TRAIN_VIDEO_PATHS, TRAIN_LABEL_PATHS, TEST_VIDEO_PATHS, TEST_LABEL_PATHS)
    model.build_graph()
    model.train(1)
