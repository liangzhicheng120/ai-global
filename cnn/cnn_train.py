#!/bin/bash
# -*- coding:utf-8 -*-
import os
import sys
from cnn.cnn_model import *
from w2v.word2vec import *
from util.label_util import *
from util.image_util import *
import config as  con
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from util.pickle_util import *
from util.label_util import *


class CnnTrain(CNNModel, Word2Vec, PickleUtil, ImageUtil, LabelUtil):
    def __init__(self):
        self.batch_size = 16
        self.imageId_caption_dict = self.pickle_load(con.pick['imageId_caption_dict'])
        self.load_W2V_Model(con.model['w2v_size200'])
        self._X = tf.placeholder(tf.float32, [None, con.image['height'], con.image['width'], con.image['dimension']])
        self._Y = tf.placeholder(tf.float32, [None, con.model['size']])
        self._keep_prob = tf.placeholder(tf.float32)
        self.x = []
        self.y = []
        self.distence = []
        pass

    def run(self):
        print('batch_size:{0}'.format(self.batch_size))
        self.train_crack_captcha_cnn()
        pass

    def plot(self, step, loss, distence):
        self.x.append(step)
        self.y.append(loss)
        self.distence.append(distence)
        plt.axis([step - 100, step, 0, 12])
        plt.plot(self.x, self.y, "r")
        plt.plot(self.x, self.distence, "b")
        plt.pause(0.01)
        pass

    def get_next_batch(self, image_ids_splist, n):
        image_ids = image_ids_splist[n]
        batch_size = len(image_ids)
        batch_x = np.zeros([batch_size, con.image['height'], con.image['width'], con.image['dimension']])
        batch_y = np.zeros([batch_size, con.model['size']])
        for i, image_id in enumerate(image_ids):
            batch_x[i, :, :, :] = self.image_to_matrix(con.image['all'] + image_id)
            batch_y[i, :] = sum(self.sentence_2_vec(self.imageId_caption_dict[image_id])) / 5
        return batch_x, batch_y

    @print_func_time
    def train_crack_captcha_cnn(self):
        '''
        训练
        :return:
        '''
        _step = 0
        plt.ion()
        _output = self.crack_captcha_cnn(X=self._X, keep_prob=self._keep_prob)
        # image_ids = self.get_file_name(con.image['1w'], 'jpg', ex_size=10000)
        image_ids = self.imageId_caption_dict.keys()
        image_ids_splist, image_ids_splist_size = self.splist(image_ids, splist_size=self.batch_size)
        # loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=_output, targets=self._Y, pos_weight=0.000001))
        loss = tf.reduce_mean(0.5 * tf.square(self._Y - tf.nn.sigmoid(_output)), axis=1)
        # loss = tf.reduce_mean(0.5 * tf.square(tf.nn.sigmoid(self._Y) - _output), axis=1)
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=_output, labels=self._Y))
        predict = tf.reshape(_output, [-1, con.model['size']])
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0005).minimize(loss)
        # saver = tf.train.Saver() #保存模型
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            while True:
                batch_x, batch_y = self.get_next_batch(image_ids_splist, _step)
                _, loss_ = sess.run([optimizer, loss], feed_dict={self._X: batch_x, self._Y: batch_y, self._keep_prob: 0.75})
                _predict = sess.run(predict, feed_dict={self._X: batch_x, self._Y: batch_y, self._keep_prob: 0.75})
                _distence = sum((np.sum((batch_y - (1/(1+np.exp(-_predict)))) ** 2, axis=1)) ** 0.5) / self.batch_size
                # _distence = sum((np.sum((batch_y - (-np.log(-1 + 1 / _predict))) ** 2, axis=1)) ** 0.5) / self.batch_size
                # self.plot(_step, loss_, _distence)
                _step += 1
                _step = 0 if _step == image_ids_splist_size - 1 else _step
                print('_step:{0} _loss:{1} _distence:{2}'.format(_step, loss_, _distence))
                # if _distence < 10:
                #     break


if __name__ == '__main__':
    cnnTrain = CnnTrain()
    cnnTrain.run()
