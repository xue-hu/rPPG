__author__ = 'Iris'

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import os
import tensorflow as tf
from IPython.display import YouTubeVideo

##loading datasets
frame_lvl_record = "D:/PycharmsProject/yutube8M/data/frame/train00.tfrecord"
video_lvl_record = "D:/PycharmsProject/yutube8M/data/video/train00.tfrecord"

##processing video-level data
vid_ids = []
labels = []
mean_rgb = []
mean_audio = []
for example in tf.python_io.tf_record_iterator(video_lvl_record):
    tf_example = tf.train.Example.FromString(example)
    vid_ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
    labels.append(tf_example.features.feature['labels'].int64_list.value)
    mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
    mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)
print('Number of videos in this tfrecord: ',len(mean_rgb))
print('Picking a youtube video id:',vid_ids[13])
print('First 20 features of a youtube video (',vid_ids[13],'):')
print(mean_rgb[13][:20])

##processing frame-level data
feat_rgb = []
feat_audio = []

for example in tf.python_io.tf_record_iterator(frame_lvl_record):
    tf_seq_example = tf.train.SequenceExample.FromString(example)
    n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)
    sess = tf.InteractiveSession()
    rgb_frame = []
    audio_frame = []
    # iterate through frames
    for i in range(n_frames):
        rgb_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8)
                       ,tf.float32).eval())
        audio_frame.append(tf.cast(tf.decode_raw(
                tf_seq_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0],tf.uint8)
                       ,tf.float32).eval())


    sess.close()
    feat_rgb.append(rgb_frame)
    feat_audio.append(audio_frame)
    break
print('The first video has %d frames' %len(feat_rgb[0]))

##explore the labels
labels_2018 = pd.read_csv('D:/PycharmsProject/yutube8M/data/label_names_2018.csv')
print("we have {} unique labels in the dataset".format(len(labels_2018['label_name'].unique())))
n=10
from collections import Counter
label_mapping = pd.read_csv('D:/PycharmsProject/yutube8M/data/label_names_2018.csv',header=0,index_col=0,squeeze=True).T.to_dict()
top_n = Counter([item for sublist in labels for item in sublist]).most_common(n)
top_n_labels = [int(i[0]) for i in top_n]
top_n_label_names = [label_mapping[x] for x in top_n_labels]
print(top_n_label_names)
