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
import random

# train label and image
scene_train_annotations_20170904 = 'E:\\LearningDeepData\\ai_challenger_scene_train_20170904\\scene_train_annotations_80_16.json'
scene_train_images_20170904 = 'E:\\LearningDeepBatch\\80_16\\{0}'

# scene_train_annotations_20170904 = 'E:\\LearningDeepData\\ai_challenger_scene_train_20170904\\scene_train_annotations_54355.json'
# scene_train_images_20170904 = 'E:\\LearningDeepData\\ai_challenger_scene_train_20170904\\scene_train_images_20170904\\{0}'

# validation label and image
scene_validation_annotations_20170908 = 'E:\\LearningDeepData\\ai_challenger_scene_validation_20170908\\scene_validation_annotations_20170908.json'
ai_challenger_scene_validation_20170908 = 'E:\\LearningDeepData\\ai_challenger_scene_validation_20170908\\scene_validation_images_20170908\\{0}'

# test label and image
scene_test_a_images_20170922 = 'E:\\LearningDeepData\\ai_challenger_scene_test_a_20170922\\scene_test_a_images_20170922\\{0}'

# 训练模型输出路径
save_model_path = 'E:\\LearningDeep\\model-vgg\\{0}'

tf.set_random_seed(1)
json_file = open(scene_train_annotations_20170904, "r")
json_data = json.load(json_file)
json_file.close()

bb = []
for i in range(len(json_data)):
    json_data[i]['new_id'] = con.label_map_converse[json_data[i]['label_id']]
    bb.append(json_data[i])
json_data = bb

validation_json_file = open(scene_validation_annotations_20170908)
validation_json_data = json.load(validation_json_file)
validation_json_file.close()

aa = []
for i in range(len(validation_json_data)):
    validation_json_data[i]['new_id'] = con.label_map_converse[validation_json_data[i]['label_id']]
    aa.append(validation_json_data[i])
validation_json_data = aa

cc = {}
for i in validation_json_data:
    image_id = i['image_id']
    label_id = str(i['label_id'])
    cc[label_id] = cc.get(label_id, '') + ',' + image_id

imageUtil = ImageUtil()

batch_size = 64
image_hight = con.image['height']
image_width = con.image['width']
n_hidden_unis = 1024
channel = 3
out_times = 3
n_classes = 80
classes = [15, 43, 22]
_x = tf.placeholder(tf.float32, [None, image_hight, image_width, channel])
_y = tf.placeholder(tf.float32, [None, n_classes])
size = 400
acc = 0.8

print('validate:{0}'.format(len(validation_json_data)))

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
    net = {}
    # 将占位符 转换为 按照图片给的新样式
    _keep_prob = 0.9
    x = tf.reshape(_x, shape=[-1, image_hight, image_width, channel])

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, channel, 32]))  # 从正太分布输出随机值
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    net['conv1'] = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    net['conv1_maxpool'] = tf.nn.max_pool(net['conv1'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    net['conv1_lrn'] = tf.nn.lrn(net['conv1_maxpool'], depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    net['conv1_dropout'] = tf.nn.dropout(net['conv1_lrn'], _keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    net['conv2'] = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(net['conv1_dropout'], w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    net['conv2_maxpool'] = tf.nn.max_pool(net['conv2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    net['conv2_lrn'] = tf.nn.lrn(net['conv2_maxpool'], depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    net['conv2_dropout'] = tf.nn.dropout(net['conv2_lrn'], _keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
    net['conv3'] = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(net['conv2_dropout'], w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    net['conv3_maxpool'] = tf.nn.max_pool(net['conv3'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    net['conv3_lrn'] = tf.nn.lrn(net['conv3_maxpool'], depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    net['conv3_dropout'] = tf.nn.dropout(net['conv3_lrn'], _keep_prob)

    w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 128]))
    b_c4 = tf.Variable(b_alpha * tf.random_normal([128]))
    net['conv4'] = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(net['conv3_dropout'], w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))
    net['conv4_maxpool'] = tf.nn.max_pool(net['conv4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    net['conv4_lrn'] = tf.nn.lrn(net['conv4_maxpool'], depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    net['conv4_dropout'] = tf.nn.dropout(net['conv4_lrn'], _keep_prob)

    w_c5 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 128]))
    b_c5 = tf.Variable(b_alpha * tf.random_normal([128]))
    net['conv5'] = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(net['conv4_dropout'], w_c5, strides=[1, 1, 1, 1], padding='SAME'), b_c5))
    net['conv5_maxpool'] = tf.nn.max_pool(net['conv5'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    net['conv5_lrn'] = tf.nn.lrn(net['conv5_maxpool'], depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    net['conv5_dropout'] = tf.nn.dropout(net['conv5_lrn'], _keep_prob)

    # TODO 可配置
    w_fc1_a, w_fc1_b, w_fc1_c = map(int, str(net['conv2_dropout'].get_shape()).replace(')', '').split(',')[1:])
    w_fc2_a, w_fc2_b, w_fc2_c = map(int, str(net['conv4_dropout'].get_shape()).replace(')', '').split(',')[1:])
    w_fc3_a, w_fc3_b, w_fc3_c = map(int, str(net['conv5_dropout'].get_shape()).replace(')', '').split(',')[1:])
    w_d1 = tf.Variable(w_alpha * tf.random_normal([w_fc1_a * w_fc1_b * w_fc1_c, 512]))
    b_d1 = tf.Variable(b_alpha * tf.random_normal([512]))
    w_d2 = tf.Variable(w_alpha * tf.random_normal([w_fc2_a * w_fc2_b * w_fc2_c, 512]))
    b_d2 = tf.Variable(b_alpha * tf.random_normal([512]))
    w_d3 = tf.Variable(w_alpha * tf.random_normal([w_fc3_a * w_fc3_b * w_fc3_c, 512]))
    b_d3 = tf.Variable(b_alpha * tf.random_normal([512]))
    net['dense1'] = tf.reshape(net['conv2_dropout'], [-1, w_d1.get_shape().as_list()[0]])
    net['dense2'] = tf.reshape(net['conv4_dropout'], [-1, w_d2.get_shape().as_list()[0]])
    net['dense3'] = tf.reshape(net['conv5_dropout'], [-1, w_d3.get_shape().as_list()[0]])

    net['dense1'] = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(net['dense1'], w_d1), b_d1)), _keep_prob)
    net['dense2'] = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(net['dense2'], w_d2), b_d2)), _keep_prob)
    net['dense3'] = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(net['dense3'], w_d3), b_d3)), _keep_prob)

    w_out1 = tf.Variable(w_alpha * tf.random_normal([512, classes[0]]))
    b_out1 = tf.Variable(b_alpha * tf.random_normal([classes[0]]))
    net['out1'] = tf.add(tf.matmul(net['dense1'], w_out1), b_out1)

    w_out2 = tf.Variable(w_alpha * tf.random_normal([512, classes[1]]))
    b_out2 = tf.Variable(b_alpha * tf.random_normal([classes[1]]))
    net['out2'] = tf.add(tf.matmul(net['dense2'], w_out2), b_out2)

    w_out3 = tf.Variable(w_alpha * tf.random_normal([512, classes[2]]))
    b_out3 = tf.Variable(b_alpha * tf.random_normal([classes[2]]))
    net['out3'] = tf.add(tf.matmul(net['dense3'], w_out3), b_out3)

    net['out'] = tf.concat([net['out1'], net['out2'], net['out3']], axis=1)

    return net['out']


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
        imgae_label = json_data[start_point + i]['new_id']
        batch_xs[i] = imageUtil.image_to_matrix(scene_train_images_20170904.format(image_name))
        batch_ys[i, eval(imgae_label)] = 1
    return batch_xs, batch_ys


# def random_get_batch():
#     start_point = random.randint(0, (len(validation_json_data) - size))
#     batch_xs = np.zeros([size, image_hight, image_width, channel])
#     batch_ys = np.zeros([size, n_classes])
#     for i in range(size):
#         image_name = validation_json_data[start_point + i]['image_id']
#         imgae_label = validation_json_data[start_point + i]['label_id']
#         batch_xs[i] = imageUtil.image_to_matrix(ai_challenger_scene_validation_20170908.format(image_name))
#         batch_ys[i][eval(imgae_label)] = 1
#     return batch_xs, batch_ys

def random_get_batch():
    dd = []
    for key, value in cc.items():
        image_ids = random.sample(value.split(',')[1:], 5)
        for image_id in image_ids:
            item = {}
            item['label_id'] = str(key)
            item['image_id'] = image_id
            item['new_id'] = con.label_map_converse[str(key)]
            dd.append(item)
    validation_json_data = dd
    batch_xs = np.zeros([size, image_hight, image_width, channel])
    batch_ys = np.zeros([size, n_classes])
    for i in range(size):
        image_name = validation_json_data[i]['image_id']
        imgae_label = validation_json_data[i]['new_id']
        batch_xs[i] = imageUtil.image_to_matrix(ai_challenger_scene_validation_20170908.format(image_name))
        batch_ys[i][eval(imgae_label)] = 1
    return batch_xs, batch_ys


def mkdir(acc, num):
    '''
    创建路径,增加精度
    :param acc:
    :return:
    '''
    path = save_model_path.format(round(acc, 3))
    os.makedirs(path)
    return path + '\\crack_image_{0}.model'.format(num)


if __name__ == '__main__':
    t0 = time.clock()
    pred = crack_captcha_cnn()
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=_y))
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    _step = 0
    _num = 1
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('batch_size:{0}'.format(batch_size))
        total = (len(json_data) / batch_size - 1)
        while True:
            if _step > total:
                _step = 0
                _num += 1
            if _num > 10:
                break
            batch_xs, batch_ys = get_next_batch_rnn(_step)
            Pred, _ = sess.run([pred, train_op], feed_dict={_x: batch_xs, _y: batch_ys})
            loss = sess.run(cost, feed_dict={_x: batch_xs, _y: batch_ys})
            if _step % 200 == 0:
                test_index_max3 = np.zeros([size, out_times])
                Pred_index_max3 = np.zeros([size, out_times])
                batch_test_xs, batch_test_ys = random_get_batch()
                Pred_test = sess.run(tf.nn.softmax(pred), feed_dict={_x: batch_test_xs, _y: batch_test_ys})
                class1 = Pred_test[:, 0:classes[0]]
                class2 = Pred_test[:, classes[0]:classes[0] + classes[1]]
                class3 = Pred_test[:, classes[0] + classes[1]:]

                test_index = bottleneck.argpartsort(-batch_test_ys, 1, axis=1)[:, :1]
                test_index_max3[:][:] = test_index
                Pred_index_max3[:, 0] = np.argmax(class1, axis=1)
                # TODO 平移
                Pred_index_max3[:, 1] = np.argmax(class2, axis=1) + classes[0]
                Pred_index_max3[:, 2] = np.argmax(class3, axis=1) + (classes[0] + classes[1])

                test_acc = np.amax(1 * np.equal(Pred_index_max3, test_index_max3), axis=1)
                test_acc = 1.0 * sum(test_acc) / len(test_acc)
                text = "_num:{0} _step:{1} _loss:{2} _accuracy:{3}".format(_num, _step, loss, test_acc)
                print(text)
            if test_acc >= acc:
                saver.save(sess, save_path=mkdir(acc, _num), global_step=_step)
                acc += 0.02
            _step += 1
        t1 = time.clock()

    print(t1 - t0)
    pass
