#!/bin/bash
# -*- coding:utf-8 -*-
import tensorflow as tf
from util.image_util import *
from util.config_util import *
from util.json_util import *
from util.file_util import *
from util.log_util import *
import json
import sys
import shutil
from sys import argv

CONFIG = read_conf_file('./config/classification.yml')

imageUtil = ImageUtil(CONFIG)
jsonUtil = JsonUtil()
fileUtil = FileUtil()

# ---------加载参数---------- #
IMAGE_PATH = CONFIG.image_path
IMAGE_HIGHT = CONFIG.small_height
IMAGE_WIDTH = CONFIG.small_width
CHANNEL = CONFIG.channal
BATCH_SIZE = CONFIG.batch_size
N_CLASSES = CONFIG.n_classes
CUT_SIZE = CONFIG.cut_size
LOSS_MODEL_FILE = CONFIG.loss_model_file
LOSS_MODEL_FILE_PATH = CONFIG.loss_model_file_path
OUTPUT_IMAGE_PATH = CONFIG.output_image_path
JSON_FILE_PATH = CONFIG.json_file_path

_x = tf.placeholder(tf.float32, [None, IMAGE_HIGHT, IMAGE_WIDTH, CHANNEL])
_y = tf.placeholder(tf.float32, [None, N_CLASSES])

net = {}
weights = {}
biases = {}
json_data = []


## ------------------单独做数据增强-------------------------- ##

def init_data():
    tf.logging.info('Start init data by data enhancement strategy')
    train_data = jsonUtil.read_json_file(JSON_FILE_PATH)
    result = []
    for i in range(len(train_data)):
        image_name = train_data[i]['image_id']
        image_matrixs = imageUtil.image_to_matrix_cut(IMAGE_PATH.format(image_name), CUT_SIZE)
        for id, image_matrix in enumerate(image_matrixs):
            image_id = str(id) + '_' + image_name
            imageUtil.matrix_to_image(image_matrix, OUTPUT_IMAGE_PATH.format(image_id))
            item = {}
            item['image_id'] = image_id
            item['label_id'] = train_data[i]['label_id']
            result.append(item)
    return result


## ------------------单独做数据增强-------------------------- ##

def get_next_batch(n, json_data):
    start_point = BATCH_SIZE * n
    batch_xs = np.zeros([BATCH_SIZE, IMAGE_HIGHT, IMAGE_WIDTH, CHANNEL])
    batch_ys = np.zeros([BATCH_SIZE, N_CLASSES])
    for i in range(BATCH_SIZE):
        image_name = json_data[start_point + i]['image_id']
        imgae_label = json_data[start_point + i]['label_id']
        batch_xs[i] = imageUtil.image_to_matrix(OUTPUT_IMAGE_PATH.format(image_name))
        batch_ys[i][eval(imgae_label)] = 1
    return batch_xs, batch_ys


def get_test(image_name):
    return np.reshape(imageUtil.image_to_matrix(image_name), [1, IMAGE_HIGHT, IMAGE_WIDTH, CHANNEL])


def build_net(ntype, nin, nwb=None, keep_prob=None, reshape=None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME') + nwb[1])
    elif ntype == 'avg_pool':
        return tf.nn.avg_pool(nin, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    elif ntype == 'max_pool':
        return tf.nn.max_pool(nin, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    elif ntype == 'lrn':
        return tf.nn.lrn(nin, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    elif ntype == 'dropout':
        return tf.nn.dropout(nin, keep_prob)

    elif ntype == 'reshape':
        return tf.reshape(nin, reshape)

    elif ntype == 'full_conncet':
        return tf.nn.relu(tf.add(tf.matmul(nin, nwb[0]), nwb[1]))

    elif ntype == 'full_conncet_disactivite':
        return tf.add(tf.matmul(nin, nwb[0]), nwb[1])


def set_weights(name, size, w_alpha=0.01):
    weights[name] = tf.Variable(w_alpha * tf.random_normal(size))


def set_biases(name, size, b_alpha=0.05):
    biases[name] = tf.Variable(b_alpha * tf.random_normal(size))


def get_weights(name):
    try:
        return weights[name]
    except:
        tf.logging.error("Cannot find weights for this name.")


def get_biases(name):
    try:
        return biases[name]
    except:
        tf.logging.error("Cannot find biases for this name.")


def get_weights_biases(name):
    try:
        return weights[name], biases[name]
    except:
        tf.logging.error("Cannot find weights or biases for this name.")


def full_connect_net():
    _keep_prob = 0.9
    x = tf.reshape(_x, shape=[-1, IMAGE_HIGHT * IMAGE_WIDTH * CHANNEL])
    set_weights('full_connect', [IMAGE_HIGHT * IMAGE_WIDTH * CHANNEL, 32])
    set_biases('full_connect', [32])
    net['dense'] = build_net('full_conncet', x, nwb=get_weights_biases(name='full_connect'))
    net['dense_dropout'] = build_net('dropout', net['dense'], keep_prob=_keep_prob)
    set_weights('out', [32, N_CLASSES])
    set_biases('out', [N_CLASSES])
    net['out'] = build_net('full_conncet_disactivite', net['dense_dropout'], nwb=get_weights_biases('out'))
    return net['out']


@print_func_time
def train():
    _, size = fileUtil.get_files(OUTPUT_IMAGE_PATH)
    if size < CUT_SIZE:
        json_data = init_data()
        json_data = random.sample(json_data, len(json_data))
    pred = full_connect_net()
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=_y))
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    _step = 0
    saver = tf.train.Saver()
    json_len = len(json_data)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while _step * BATCH_SIZE < json_len - 1:
            batch_xs, batch_ys = get_next_batch(_step, json_data)
            Cost, _ = sess.run([cost, train_op], feed_dict={_x: batch_xs, _y: batch_ys})
            tf.logging.info("step: {0}, cost: {1}".format(_step, Cost))
            _step += 1
        if _step >= 1:
            saver.save(sess, save_path=LOSS_MODEL_FILE_PATH, global_step=_step)
            tf.logging.info('The model save path:{0}'.format(LOSS_MODEL_FILE_PATH))


@print_func_time
def test(pic):
    pred = full_connect_net()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, CONFIG.loss_model_file)
        batch_test_xs = get_test(pic)
        Pred = sess.run(tf.nn.softmax(pred), feed_dict={_x: batch_test_xs})
        tf.logging.info('pred:{0}'.format(Pred))


if __name__ == "__main__":
    tf.logging.set_verbosity('INFO')
    try:
        model = argv[1]
        if model == 'train':
            train()
        if model == 'test':
            try:
                image_name = argv[2]
                test(image_name)
            except Exception:
                tf.logging.warn('Test model should be have a image name.')
    except Exception:
        tf.logging.warn('Model can not be null.')
