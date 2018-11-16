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
TRAIN_LABEL_PATHS = ['D:\PycharmsProject\yutube8M\data\ple.txt']
TEST_VIDEO_PATHS = ['D:\PycharmsProject\yutube8M\data\Logitech HD Pro Webcam C920.avi']
TEST_LABEL_PATHS = ['D:\PycharmsProject\yutube8M\data\ple.txt']


class VideoAnalysis(object):
    def __init__(self, train_video_paths, train_label_paths,
                 test_video_paths, test_label_paths,
                 img_width=492, img_height=492):
        self.train_video_paths = train_video_paths
        self.train_label_paths = train_label_paths
        self.test_video_paths = test_video_paths
        self.test_label_paths = test_label_paths
        self.width = img_width
        self.height = img_height
        self.lr = 0.1
        self.batch_size = 16
        self.gstep = tf.Variable(0, trainable=False, name='global_step')
        self.skip_step = 1

    def get_data(self, video_paths, label_paths):
        print("create generator....")
        frame_gen = process_data.crop_resize_face(video_paths, self.width, self.height)
        diff_gen = process_data.nor_diff_face(video_paths, self.width, self.height)
        sample_gen = process_data.get_sample(frame_gen, diff_gen, label_paths)
        batch_gen = process_data.get_batch(sample_gen, self.batch_size)
        return batch_gen

    # def create_training_input(self, train_video_path, train_label_path):
    #     print("create new training-set generator.....")
    #     self.train_gen = self.get_data(train_video_path, train_label_path)
    #
    # def create_testing_input(self, test_video_path, test_label_path):
    #     print("create new testing-set generator.....")
    #     self.test_gen = self.get_data(test_video_path, test_label_path)

    def loading_model(self):
        with tf.name_scope('Input'):
            self.input_img = tf.placeholder(dtype=tf.float32, name='input_img',
                                        shape=[self.batch_size, self.width, self.height, 3])
            self.input_diff = tf.placeholder(dtype=tf.float32, name='input_diff',
                                         shape=[self.batch_size, self.width, self.height, 3])
        self.model = loading_model.NnModel(self.input_img, self.input_diff)
        #self.model.vgg_load()
        self.model.two_stream_vgg_load()

    def inference(self):
        out = self.model.d_fc7
        #############create pred###########################
        with tf.variable_scope('fc8', reuse=tf.AUTO_REUSE) as scope:
            w = tf.get_variable("weight", dtype=tf.float32, initializer=tf.random_normal(out.shape))
            b = tf.get_variable("bias", dtype=tf.float32, initializer=tf.zeros([self.batch_size, ]))
        z = tf.reduce_sum(tf.multiply(out, w)) + b
        self.pred = tf.nn.relu(z, name=scope.name)

    ################################################

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
        for tr_v, tr_l in zip(self.train_video_paths, self.train_label_paths):
            train_gen = self.get_data(tr_v, tr_l)
            try:
                while True:
                    print("epoch " + str(epoch + 1) + "-" + str(n_batch + 1))
                    frames, diffs, labels = next(train_gen)
                    ##########testing the generator##############
                    # for frame, diff, label in zip(frames, diffs, labels):
                    #     cv2.imshow('face', frame)
                    #     cv2.imshow('diff', diff)
                    #     print(label)
                    #     cv2.waitKey(0)
                    ############################################
                    loss, _, summary = sess.run([self.loss, self.opt, summary_op], feed_dict={self.input_img: frames,
                                                                                              self.input_diff: diffs,
                                                                                              self.labels: labels})
                    total_loss += loss
                    n_batch += 1
                    writer.add_summary(summary, global_step=step)
                    step += 1
                    print('Average loss at batch {0}: {1}'.format(n_batch, total_loss / n_batch))
            except tf.errors.OutOfRangeError:
                pass
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batch))
        print('Took:{0} seconds'.format(time.time() - start_time))
        if epoch % self.skip_step == 0:
            #writer.add_summary(summary, global_step=step)
            saver.save(sess, 'checkpoint_dict/VideoAnalysis', global_step=self.gstep)
        return step

    def eval_once(self, sess, writer, summary_op, epoch, step):
        print("begin to evaluate.....")
        total_accuracy = 0
        n_test = 0
        for te_v, te_l in zip(self.test_video_paths, self.test_label_paths):
            test_gen = self.get_data(te_v, te_l)
            try:
                while True:
                    frames, diffs, labels = next(test_gen)
                    accuracy, summary = sess.run([self.accuracy, summary_op],
                                                 feed_dict={self.input_img: frames,
                                                            self.input_diff: diffs, self.labels: labels})
                    total_accuracy += accuracy
                    n_test += 1
                    writer.add_summary(summary, global_step=step)
            except tf.errors.OutOfRangeError:
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
    train_prob_ids= [2,4]
    test_prob_ids = [3]
    tr_vd_paths, tr_lb_paths = utils.create_file_paths(train_prob_ids)
    te_vd_paths, te_lb_paths = utils.create_file_paths(test_prob_ids)
    model = VideoAnalysis(tr_vd_paths, tr_lb_paths, te_vd_paths, te_lb_paths)
    ######################################################################################
    #model = VideoAnalysis(TRAIN_VIDEO_PATHS, TRAIN_LABEL_PATHS, TEST_VIDEO_PATHS, TEST_LABEL_PATHS)
    model.build_graph()
    model.train(10)
