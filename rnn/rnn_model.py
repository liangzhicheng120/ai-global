# coding:utf-8
import sys

sys.path.append("..")
import tensorflow as tf
from numpy import random
import numpy as np
from util.label_util import *
from util.image_util import *
from util.code_util import *
from util.pickle_util import *
import pickle

tf.set_random_seed(1)

# 映射关系
dictionary_words = pickle.load(open("E:/LearningDeepData/chinese_dictionary_3001_index.pik", "rb"))
dictionary_index = pickle.load(open("E:/LearningDeepData/chinese_dictionary_3001_word.pik", "rb"))
# str = index2str[500]
# aa = ["a".decode('utf-8'), "b".decode('utf-8'), "c".decode('utf-8')]
# index = str2index["a".decode('utf-8'), "b".decode('utf-8')]
# # print(str)
# print(index)

# hyperparameters
training_iters = 2100000
batch_size = 16

channel = 3
image_hight = 256  # shape 28*28   @image_hight = 256
image_width = 256  # time steps     @image_width = 256
char_size = 12  # con.char['bit_size']  # classes 0-9  @char_size = 20
char_len = 50
n_hidden_unis = char_len + 1  # neurons in hidden layer  @max_char = 50

# tf Graph input
x = tf.placeholder(tf.float32, [None, char_len, channel * image_width * image_hight])
y = tf.placeholder(tf.float32, [None, char_size * char_len])

_x = tf.placeholder(tf.float32, [None, image_hight, image_width, channel])

# Define weights
weights = {
    # (256,50)
    'in': tf.Variable(tf.random_normal([char_size, char_size])),
    # (128,10)
    'out': [tf.Variable(tf.random_normal([char_size * char_len, char_len])), tf.Variable(tf.random_normal([char_len, char_size * char_len]))]
}
biases = {
    # (128,)
    'in': tf.Variable(tf.constant(0.1, shape=[char_size, ])),
    # (10,)
    'out': [tf.Variable(tf.constant(0.1, shape=[char_len, ])), tf.Variable(tf.constant(0.1, shape=[char_size * char_len, ]))]
}

########### 初始化工具类 ######
labelUtil = LabelUtil()
imageUtil = ImageUtil()
codeUtil = CodeUtil()
pickleUtil = PickleUtil()

########### 加载文件 ############
imageId_caption_dict = pickleUtil.pickle_load(con.pick['imageId_caption_dict'])
image_ids = imageId_caption_dict.keys()
image_ids_splist, image_ids_splist_size = labelUtil.splist(image_ids, splist_size=batch_size)


def word2index(str):
    index = dictionary_index[str]
    return index
    pass


def index2word(index):
    word = dictionary_words.get(index)
    return word if word else ''


# TODO 改
def str_to_format_matrix(image_id):
    # [2979, 1463, 2736, 1092, 349, 1787, 1870, 74, 1702, 2426, 2488, 835, 1092, 2979, 1463, 2736, 1092, 1156, 74, 1702, 2198, 2488, 2656, 1037, 2403, 253, 2896, 1037]

    matrix = np.zeros([char_len, char_size])
    sen = imageId_caption_dict[image_id][0].replace("\n", "")
    index_list = list(map(word2index, sen))

    end_list = (list(codeUtil.num_to_bin(dictionary_index['。'])))

    end_array = np.array(list(map(int, end_list)))

    index_bin_list = list(map(codeUtil.num_to_bin, index_list))
    index_bin_list = list(map(lambda x: list(map(eval, x)), index_bin_list))
    index_bin_matrix = np.array(index_bin_list)

    matrix[0:len(index_bin_matrix), :] = index_bin_matrix
    matrix[len(index_bin_matrix):, :] = end_array
    return matrix


def crack_captcha_cnn(w_alpha=0.05, b_alpha=0.2):
    '''
    define Convolutional Neural Networks
    :param w_alpha:
    :param b_alpha:
    :return:
    '''
    # 将占位符 转换为 按照图片给的新样式
    _keep_prob = 0.9
    x = tf.reshape(_x, shape=[-1, image_hight, image_width, 3])

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
    #
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, _keep_prob)

    # w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    # b_c4 = tf.Variable(b_alpha * tf.random_normal([128]))
    # conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
    # conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv4 = tf.nn.dropout(conv4, _keep_prob)

    # w_c5 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 128]))
    # b_c5 = tf.Variable(b_alpha * tf.random_normal([128]))
    # conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4, w_c5, strides=[1, 1, 1, 1], padding='SAME'), b_c5))
    # conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv5 = tf.nn.dropout(conv5, _keep_prob)

    # TODO 可配置
    in_conv = conv3
    # Fully connected layer
    w_a, w_b, w_c = map(int, str(in_conv.get_shape()).replace(')', '').split(',')[1:])
    w_d = tf.Variable(w_alpha * tf.random_normal([w_a * w_b * w_c, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(in_conv, [-1, w_d.get_shape().as_list()[0]])

    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, _keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, char_size]))
    b_out = tf.Variable(b_alpha * tf.random_normal([char_size]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


def RNN(X, weights, biases):
    # hidden layer for input to cell
    # X(128 batch, 256 steps, 256 inputs) => (batch_size * n_hidden_unis, 3001)
    X = tf.reshape(X, [-1, char_size])
    # ==>(128 batch * 28 steps, 28 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [batch_size, -1, n_hidden_unis])
    # ==>(128 batch , 28 steps, 28 hidden)

    # cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts(c_state, m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    re_output = outputs[:, :, :char_len]
    re_output = tf.reshape(re_output, [-1, char_size * char_len])

    # hidden layer for output as the final results
    re_output = tf.nn.relu(tf.matmul(re_output, weights['out'][0]) + biases['out'][0])  # states[1]->m_state states[1]=output[-1]
    results = tf.matmul(re_output, weights['out'][1]) + biases['out'][1]
    # outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results, outputs, states


def CNN_and_RNN():
    cnn_out = crack_captcha_cnn()
    cnn_out = tf.reshape(cnn_out, [-1, 1, char_size])
    x_rnn = tf.reshape(y, [-1, char_len, char_size])
    x_rnn = tf.concat([cnn_out, x_rnn], 1)
    print(np.shape(x_rnn))
    pred, outputs, states = RNN(x_rnn, weights, biases)
    return pred, outputs, states


def get_next_batch_rnn(image_ids_splist, n):
    image_ids = image_ids_splist[n]
    batch_size = len(image_ids)
    batch_xs = np.zeros([batch_size, image_hight, image_width, channel])
    batch_ys = np.zeros([batch_size, char_len * char_size])
    for i, image_id in enumerate(image_ids):
        # batch_xs[i, :, :] = np.hstack((matrix_channel3[:, :, 0], matrix_channel3[:, :, 1], matrix_channel3[:, :, 2]))
        batch_xs[i, :, :, :] = imageUtil.image_to_matrix(con.image['all'] + image_id)
        batch_ys[i, :] = str_to_format_matrix(image_id).flatten()
    return batch_xs, batch_ys


def int_list(source_list):
    return list(map(int, source_list))


def join_list(source_list):
    return list(map(lambda x: int(''.join(str(x))), source_list))


if __name__ == '__main__':

    # pred, outputs, states = RNN(x, weights, biases)
    pred, outputs, states = CNN_and_RNN()
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
    # correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    _step = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            batch_xs, batch_ys = get_next_batch_rnn(image_ids_splist, _step)
            # Outputs = sess.run(outputs, feed_dict={_x: batch_xs, y: batch_ys})
            Pred = sess.run(pred, feed_dict={_x: batch_xs, y: batch_ys})
            # States = sess.run(states, feed_dict={_x: batch_xs, y: batch_ys})
            # print("Outputs shape: {0}".format(np.shape(Outputs)))
            # print("Pred shape: {0}".format(np.shape(Pred)))
            predict_mix = np.around(1.0 / (1 + np.exp(Pred)))
            predict_mix = np.reshape(predict_mix, [-1, char_len, char_size])
            # print("hahahah{0}".format(np.shape(predict_mix)))
            # print("States shape: {0}".format(np.shape(States)))

            sess.run([train_op], feed_dict={_x: batch_xs, y: batch_ys})
            if _step % 20 == 0:
                for i in range(batch_size):
                    predict_list = list(map(int_list, predict_mix[i].tolist()))
                    predict_list = list(map(lambda x: ''.join(map(str, x)), predict_list))
                    index_list = list(map(codeUtil.bin_to_dec, predict_list))
                    print(index_list)
                    # index_list = np.argmax(predict_mix[i], axis=1)
                    # print(index_list)
                    try:
                        str_list = map(index2word, index_list)
                        sen = ''.join(str_list)
                        print(sen)
                    except Exception:
                        print('unexcept word')
                        # str = codeUtil.matrix_to_str(predict_mix[i])
                        # try:
                        #     print(str)
                        # except:
                        #     print('utf-8 cant encode')
            _step += 1
            _step = 0 if _step == image_ids_splist_size - 1 else _step
            print("_step:{0} _loss:{1}".format(_step, sess.run(cost, feed_dict={_x: batch_xs, y: batch_ys})))
