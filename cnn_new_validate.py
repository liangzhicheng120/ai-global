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
from sklearn.naive_bayes import GaussianNB
from collections import Counter

# validation label and image
scene_validation_annotations_20170908 = 'E:\\LearningDeepData\\ai_challenger_scene_validation_20170908\\scene_validation_annotations_20170908.json'
ai_challenger_scene_validation_20170908 = 'E:\\LearningDeepData\\ai_challenger_scene_validation_20170908\\scene_validation_images_20170908\\{0}'

tf.set_random_seed(1)

validation_json_file = open(scene_validation_annotations_20170908)
validation_json_data = json.load(validation_json_file)
validation_json_file.close()

imageUtil = ImageUtil()
image_hight = con.image['height']
image_width = con.image['width']
n_hidden_unis = 1024
channel = 3
out_times = 3
n_classes = 80
_x = tf.placeholder(tf.float32, [None, image_hight, image_width, channel])
_y = tf.placeholder(tf.float32, [None, n_classes])

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
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 3, 32]))  # 从正太分布输出随机值
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, _keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, _keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, _keep_prob)

    w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    b_c4 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv4 = tf.nn.dropout(conv4, _keep_prob)

    w_c5 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 128]))
    b_c5 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4, w_c5, strides=[1, 1, 1, 1], padding='SAME'), b_c5))
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv5 = tf.nn.dropout(conv5, _keep_prob)

    # TODO 可配置
    in_conv = conv5
    # Fully connected layer
    w_a, w_b, w_c = map(int, str(in_conv.get_shape()).replace(')', '').split(',')[1:])
    w_d = tf.Variable(w_alpha * tf.random_normal([w_a * w_b * w_c, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(in_conv, [-1, w_d.get_shape().as_list()[0]])

    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, _keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, n_classes]))
    b_out = tf.Variable(b_alpha * tf.random_normal([n_classes]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


def get_next_batch_validation(n):
    batch_xs = np.zeros([1, image_hight, image_width, channel])
    batch_ys = np.zeros([1, n_classes])
    image_name = validation_json_data[n]['image_id']
    imgae_label = validation_json_data[n]['label_id']
    batch_xs[0] = imageUtil.image_to_matrix(ai_challenger_scene_validation_20170908.format(image_name))
    batch_ys[0][eval(imgae_label)] = 1

    return batch_xs, batch_ys


def count_array(arr, size):
    result = np.array(sorted([[np.sum(arr == i), i] for i in set(arr.flat)]))
    return result[-size:, 1]


def count_dict(dict, top):
    result = []
    for key, value in dict.items():
        values = value[1:].split(',')
        value_count = Counter(values)
        top_n = value_count.most_common(top)
        for i in top_n:
            result.append(i[0])
    return result


@print_func_time
def test():
    pred = crack_captcha_cnn()
    saver = tf.train.Saver()
    keep_prob = tf.placeholder(tf.float32)
    # count = 0
    # with tf.Session() as sess, open('result.txt', 'a') as f:
    #
    #     saver.restore(sess, "E:\\LearningDeep\\model\\0.70\\crack_image.model-1473")
    #     test_index_max3 = np.zeros([1, out_times])
    #     _step = 0
    #     total = 1000
    #     while _step < total:
    #         sess.run(tf.global_variables_initializer())
    #         result = {}
    #         for i in range(10):
    #             batch_test_xs, batch_test_ys = get_next_batch_validation(_step)
    #             Pred_test = sess.run(pred, feed_dict={_x: batch_test_xs})
    #             test_index = bottleneck.argpartsort(-batch_test_ys, 1, axis=1)[:, :1]
    #             test_index_max3[:][:] = test_index
    #             Pred_index_max3 = bottleneck.argpartsort(-Pred_test, out_times, axis=1)[:, :out_times]
    #             index = str(test_index.tolist()[0][0])
    #             predict = ','.join(list(map(str, Pred_index_max3.tolist()[0])))
    #             result[str(_step)] = result.get(str(_step), '') + ',' + predict
    #         _step += 1
    #         max_3 = count_dict(result, 3)
    #         if index in max_3:
    #             count += 1
    #             # print(index, max_3)
    #             # if _step % 100 == 0:
    #             # print(_step)
    #     print(count, count / total)










            #############################################################
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "E:\\LearningDeep\\model\\0.70\\crack_image.model-1473")
        test_index_max3 = np.zeros([1, out_times])
        _step = 0
        test_true_num = 0
        total = 1000
        while _step < total:
            batch_test_xs, batch_test_ys = get_next_batch_validation(_step)
            Pred_test = sess.run(pred, feed_dict={_x: batch_test_xs})
            test_index = bottleneck.argpartsort(-batch_test_ys, 1, axis=1)[:, :1]
            test_index_max3[:][:] = test_index
            Pred_index_max3 = bottleneck.argpartsort(-Pred_test, out_times, axis=1)[:, :out_times]
            test_true = np.amax(1 * np.equal(Pred_index_max3, test_index_max3), axis=1)
            test_true_num += sum(test_true)
            _step += 1
            if _step % 100 == 0:
                print(_step)

        test_acc = 1.0 * test_true_num / total
        print("test_true_num:{0} test_acc: {1}".format(test_true_num, test_acc))


if __name__ == '__main__':
    test()
