#!/bin/bash
# -*- coding=utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import sys


class SeriesPredictor:
    def __init__(self, input_dim, seq_size, hidden_dim=10):
        self.input_dim = input_dim  # A 超参数
        self.seq_size = seq_size  # A 超参数
        self.hidden_dim = hidden_dim  # A 超参数
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')  # B 权重变量和输入占位符
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')  # B 权重变量和输入占位符
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])  # B 权重变量和输入占位符
        self.y = tf.placeholder(tf.float32, [None, seq_size])  # B 权重变量和输入占位符
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))  # C 成本优化器（cost optimizer）
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)  # C 成本优化器（cost optimizer）
        self.saver = tf.train.Saver()  # D 辅助操作
        self.model_path = 'E:\\LearningDeep\\model\\rnn\\'  # E 模型存储位置
        self.train_step = 999999999999  # F 训练次数
        self.loss = 0.00009  # G loss最小值

    def model(self):
        """
        :param x: inputs of size [T, batch_size, input_size]
        :param W: matrix of fully-connected output layer weights
        :param b: vector of fully-connected output layer biases
        """
        cell = rnn.BasicLSTMCell(self.hidden_dim)  # A 创建一个LSTM单元。
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)  # B 运行输入单元，获取输出和状态的张量。
        num_examples = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])  # C 将输出层计算为完全连接的线性函数。
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        return out

    def train(self, train_x, train_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            for i in range(1, self.train_step + 1):  # A
                _, loss = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if i % 100 == 0:
                    print('step:{0} loss:{1}'.format(i, loss))
                if loss <= self.loss:
                    os.makedirs(self.model_path)
                    save_path = self.saver.save(sess, self.model_path + 'model.ckpt')
                    print('Model saved to {}'.format(save_path))
                    break

    def test(self, test_x):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            self.saver.restore(sess, self.model_path + 'model.ckpt')
            output = sess.run(self.model(), feed_dict={self.x: test_x})
            output = np.around(output)
            print(output)


if __name__ == '__main__':
    predictor = SeriesPredictor(input_dim=1, seq_size=4, hidden_dim=10)
    # train_x = [[[1], [2], [5], [6]],
    #            [[5], [7], [7], [8]],
    #            [[3], [4], [5], [7]]]
    # train_y = [[1, 3, 7, 11],
    #            [5, 12, 14, 15],
    #            [3, 7, 9, 12]]
    # predictor.train(train_x, train_y)
    #
    test_x = [[[1], [2], [3], [4]],  # A
              [[4], [5], [6], [7]]]  # B

    predictor.test(test_x)

    # A预测结果应为1，3，5，7。
    # B预测结果应为4，9，11，13。
