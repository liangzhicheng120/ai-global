# coding:utf-8
import sys
import tensorflow as tf
from numpy import random
import bottleneck
import numpy as np
import json
from util.image_util import *
import config as con
import os
import math

class_min = 0
class_max = 5

# train label and image
scene_train_annotations_20170904 = 'E:\\LearningDeepData\\ai_challenger_scene_train_20170904\\scene_train_annotations_0_5.json'
scene_train_images_20170904 = 'E:\\LearningDeepBatch\\0_5\\{0}'

# validation label and image
scene_validation_annotations_20170908 = 'E:\\LearningDeepData\\ai_challenger_scene_validation_20170908\\scene_validation_annotations_20170908.json'
ai_challenger_scene_validation_20170908 = 'E:\\LearningDeepData\\ai_challenger_scene_validation_20170908\\scene_validation_images_20170908\\{0}'

# test label and image
scene_test_a_images_20170922 = 'E:\\LearningDeepData\\ai_challenger_scene_test_a_20170922\\scene_test_a_images_20170922\\{0}'

tf.set_random_seed(1)
json_file = open(scene_train_annotations_20170904, "r")
json_data = json.load(json_file)
json_file.close()
bb = []
for i in range(len(json_data)):
    if class_min <= eval(json_data[i]['label_id']) <= class_max:
        bb.append(json_data[i])
json_data = bb

validation_json_file = open(scene_validation_annotations_20170908)
validation_json_data = json.load(validation_json_file)
validation_json_file.close()

aa = []
for i in range(len(validation_json_data)):
    if class_min <= eval(validation_json_data[i]['label_id']) <= class_max:
        aa.append(validation_json_data[i])
validation_json_data = aa

imageUtil = ImageUtil()

batch_size = 8
image_hight = con.image['height']
image_width = con.image['width']
n_hidden_unis = 1024
channel = 3
out_times = 1
n_classes = 2
_x = tf.placeholder(tf.float32, [None, image_hight, image_width, channel])
_y = tf.placeholder(tf.float32, [None, n_classes])
size = len(validation_json_data)
acc = 0.88

# Define weights
weights = {
    # (256,50)
    'in': tf.Variable(tf.random_normal([image_width * image_hight * channel, n_hidden_unis])),
    # (128,10)
    'out': [tf.Variable(tf.random_normal([n_hidden_unis, n_hidden_unis])),
            tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))]
}
biases = {
    # (128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unis, ])),
    # (10,)
    'out': [tf.Variable(tf.constant(0.1, shape=[n_hidden_unis, ])), tf.Variable(tf.constant(0.1, shape=[n_classes, ]))]
}


def crack_captcha_cnn(w_alpha=0.05, b_alpha=0.2):
    '''
    define Convolutional Neural Networks
    :param w_alpha:
    :param b_alpha:
    :return:
    '''
    # 将占位符 转换为 按照图片给的新样式
    _keep_prob = 1
    x = tf.reshape(_x, shape=[-1, image_hight, image_width, channel])

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 3, 16]))  # 从正太分布输出随机值
    b_c1 = tf.Variable(b_alpha * tf.random_normal([16]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, _keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 16, 32]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, _keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, _keep_prob)

    # w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    # b_c4 = tf.Variable(b_alpha * tf.random_normal([128]))
    # conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
    # conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv4 = tf.nn.dropout(conv4, _keep_prob)
    #
    # w_c5 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 128]))
    # b_c5 = tf.Variable(b_alpha * tf.random_normal([128]))
    # conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4, w_c5, strides=[1, 1, 1, 1], padding='SAME'), b_c5))
    # conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv5 = tf.nn.dropout(conv5, _keep_prob)

    # TODO 可配置
    in_conv = conv3
    # Fully connected layer
    w_a, w_b, w_c = map(int, str(in_conv.get_shape()).replace(')', '').split(',')[1:])
    # print(w_a, w_b, w_c)
    w_d = tf.Variable(w_alpha * tf.random_normal([w_a * w_b * w_c, 256]))
    b_d = tf.Variable(b_alpha * tf.random_normal([256]))
    dense = tf.reshape(in_conv, [-1, w_d.get_shape().as_list()[0]])

    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, _keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([256, n_classes]))
    b_out = tf.Variable(b_alpha * tf.random_normal([n_classes]))
    out = tf.add(tf.matmul(dense, w_out), b_out)

    # in_conv = conv5
    # # Fully connected layer
    # w_a, w_b, w_c = map(int, str(in_conv.get_shape()).replace(')', '').split(',')[1:])
    # w_d = tf.Variable(w_alpha * tf.random_normal([w_a * w_b * w_c, 1024]))
    # b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    # dense = tf.reshape(in_conv, [-1, w_d.get_shape().as_list()[0]])
    #
    # dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    # dense = tf.nn.dropout(dense, _keep_prob)
    #
    # w_out = tf.Variable(w_alpha * tf.random_normal([1024, n_classes]))
    # b_out = tf.Variable(b_alpha * tf.random_normal([n_classes]))
    # out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


def RNN(X):
    _keep_prob = 1
    # hidden layer for input to cell
    # X(128 batch, 256 steps, 256 inputs) => (batch_size * n_hidden_unis, 3001)
    X = tf.reshape(X, [-1, image_width * image_hight * channel])
    # ==>(128 batch * 28 steps, 28 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [batch_size, -1, n_hidden_unis])
    # ==>(128 batch , 28 steps, 28 hidden)

    # cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts(c_state, m_state)
    lstm_multi = tf.nn.rnn_cell.MultiRNNCell([lstm_cell], state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    print("shape:{0}".format(np.shape(states[1])))
    re_output = states[1]
    re_output = tf.reshape(re_output, [-1, n_hidden_unis])

    # hidden layer for output as the final results
    results = tf.nn.relu(
        tf.matmul(re_output, weights['out'][0]) + biases['out'][0])  # states[1]->m_state states[1]=output[-1]
    results = tf.nn.dropout(results, _keep_prob)
    results = tf.matmul(re_output, weights['out'][1]) + biases['out'][1]
    # outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results, outputs, states


def get_next_batch_rnn(n):
    start_point = batch_size * n
    batch_crop_xs = np.zeros([batch_size, image_hight, image_width, channel * 10])
    batch_xs = np.zeros([batch_size, image_hight, image_width, channel])
    batch_ys = np.zeros([batch_size, n_classes])
    for i in range(batch_size):
        image_name = json_data[start_point + i]['image_id']
        imgae_label = json_data[start_point + i]['label_id']
        batch_xs[i] = imageUtil.image_to_matrix(scene_train_images_20170904.format(image_name))
        batch_ys[i][eval(imgae_label) - class_min] = 1
    return batch_xs, batch_ys


def random_get_batch():
    start_point = 0
    batch_xs = np.zeros([size, image_hight, image_width, channel])
    batch_ys = np.zeros([size, class_max-class_min+1])
    for i in range(size):
        image_name = validation_json_data[start_point + i]['image_id']
        imgae_label = validation_json_data[start_point + i]['label_id']
        batch_xs[i] = imageUtil.image_to_matrix(ai_challenger_scene_validation_20170908.format(image_name))
        batch_ys[i][eval(imgae_label)] = 1
    return batch_xs, batch_ys


def mkdir(acc):
    '''
    创建路径,增加精度
    :param acc:
    :return:
    '''
    path = 'E:\\LearningDeep\\model-1101\\{0}_{1}\\{2}\\'.format(class_min, class_max, round(acc, 3))
    os.makedirs(path)
    return path + 'crack_image_{0}_{1}.model'.format(class_min, class_max)


if __name__ == '__main__':
    pred = crack_captcha_cnn()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=_y))
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    _step = 0
    _num = 1
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "E:\\LearningDeep\\model-1101\\0_1\\0.88\\crack_image_0_1.model-93")
        batch_test_xs, batch_test_ys = random_get_batch()
        Pred_test0_1 = sess.run(tf.nn.softmax(pred), feed_dict={_x: batch_test_xs})
        test_index0_1 = bottleneck.argpartsort(-batch_test_ys, 1, axis=1)[:, :1]

        saver.restore(sess, "E:\\LearningDeep\\model-1101\\2_3\\0.8\\crack_image_2_3.model-95")
        batch_test_xs, batch_test_ys = random_get_batch()
        Pred_test2_3 = sess.run(tf.nn.softmax(pred), feed_dict={_x: batch_test_xs})
        test_index2_3 = bottleneck.argpartsort(-batch_test_ys, 1, axis=1)[:, :1]

        saver.restore(sess, "E:\\LearningDeep\\model-1101\\4_5\\0.86\\crack_image_4_5.model-73")
        batch_test_xs, batch_test_ys = random_get_batch()
        Pred_test4_5 = sess.run(tf.nn.softmax(pred), feed_dict={_x: batch_test_xs})
        test_index4_5 = bottleneck.argpartsort(-batch_test_ys, 1, axis=1)[:, :1]

    Pred_test = np.append(Pred_test0_1, Pred_test2_3, axis=1)
    Pred_test = np.append(Pred_test, Pred_test4_5, axis=1)
    Pred_test_index = np.argmax(Pred_test, axis=1)

    result = sum(1 * np.equal(np.reshape(test_index0_1, [len(test_index0_1)]), Pred_test_index)) / len(Pred_test_index)
    print(result)
