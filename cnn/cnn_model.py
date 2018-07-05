#!/bin/bash
# -*- coding:utf-8 -*-
import tensorflow as tf
import config as con


class CNNModel(object):
    # 定义CNN
    def crack_captcha_cnn(self, X, keep_prob, w_alpha=0.05, b_alpha=0.2):
        '''
        定义网络结构
        :param w_alpha:
        :param b_alpha:
        :return:
        '''
        # 将占位符 转换为 按照图片给的新样式
        x = tf.reshape(X, shape=[-1, con.image['height'], con.image['width'], con.image['dimension']])

        # 4 conv layer
        w_c1 = tf.Variable(w_alpha * tf.random_normal([2, 2, 3, 32]))  # 从正太分布输出随机值
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, keep_prob)

        w_c2 = tf.Variable(w_alpha * tf.random_normal([2, 2, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, keep_prob)

        w_c3 = tf.Variable(w_alpha * tf.random_normal([2, 2, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.avg_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, keep_prob)

        # TODO 可配置
        conv = conv3
        # Fully connected layer
        w_a, w_b, w_c = map(int, str(conv.get_shape()).replace(')', '').split(',')[1:])
        w_d = tf.Variable(w_alpha * tf.random_normal([w_a * w_b * w_c, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv, [-1, w_d.get_shape().as_list()[0]])

        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob)

        w_out = tf.Variable(w_alpha * tf.random_normal([1024, con.model['size']]))
        b_out = tf.Variable(b_alpha * tf.random_normal([con.model['size']]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        return out
