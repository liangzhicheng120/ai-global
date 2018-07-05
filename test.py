#!/bin/bash
# -*-coding=utf-8-*-
import pickle
from util.log_util import *
import config as con
from util.pickle_util import *
from util.label_util import *
import json
from util.file_util import *
import numpy as np
from collections import Counter
import tensorflow as tf
import random

# if __name__ == '__main__':
# dict = {}
# count = {}
# with open('log.txt', 'r') as f:
#     for line in f:
#         act, test_1, test_2, test_3 = line.strip().split('\t')
#         dict[int(act)] = dict.get(int(act), 0) + 1
#         if act in [test_1, test_2, test_3]:
#             print(line.strip())
#             count[int(act)] = count.get(int(act), 0) + 1
# print_dict_info(dict)
# print_dict_info(count)

# count = {}
# with open(con.json['scene_train_annotations'], 'r') as f:
#     json_data = json.load(f)
#     for i in range(len(json_data)):
#         labelId = int(json_data[i]['label_id'])
#         imageId = json_data[i]['image_id']
#         count[labelId] = count.get(labelId, 0) + 1
#
# # print_dict_info(count)

# fileUtil = FileUtil()
# a, _ = fileUtil.get_files('E:\\gas')
# for line in a:
#     print(line)

# fileUtil = FileUtil()
# files, _ = fileUtil.get_files(con.image['scene_train'])
# pickleUtil = PickleUtil()
# with open(con.json['scene_train_annotations'], 'r') as f:
#     json_data = json.load(f)
#     for i in range(len(json_data)):
#         print(json_data[i])
#         break


# def count_array(arr, size):
#     result = np.array(sorted([[np.sum(arr == i), i] for i in set(arr.flat)]))
#     return result[-size:, 1]
#
#
# b = np.array([[0, 4, 4], [2, 0, 3], [1, 3, 4]])
# print('b=')
# print(b)
# l = np.array(sorted([[np.sum(b == i), i] for i in set(b.flat)]))
# '''
# np.sum(b==i) #统计b中等于i的元素个数
# set(b.flat)  #将b转为一维数组后，去除重复元素
# sorted()     #按元素个数从小到大排序
# l[-1]        #取出元素个数最多的元组对 (count,element)
# '''
# print(l)
# # print(l[-3:])
# print(l[-3:, 1])
#
# # print(l[-2:])
# result = []
# a = {'1': ',5,6,66,5,6,8,5,66,55,6,5,26,5,35,6,66,5,6,5,66,26,5,6,55,5,3,6,5,6,66'}
# for key, value in a.items():
#     values = value[1:].split(',')
#     value_count = Counter(values)
#     top_n = value_count.most_common(3)
#     for i in top_n:
#         result.append(i[0])
# print(result)

# cd到Anaconda下的Scripts目录
# cd Program Files\Anaconda3\Scripts

# 输入
# pip install tensorflow

# 一键安装CPU、GPU版
# pip install --upgrade --ignore-installed tensorflow
# pip install --upgrade --ignore-installed tensorflow-gpu

# 测试,cmd输入
# python
# import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))

# 输出：Hello, TensorFlow!



# from captcha.image import ImageCaptcha
# import matplotlib.pyplot as plt
# import numpy as np
# import random
# import string
# import pylab
# import matplotlib
#
# # 设置显示中文
# # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# # characters = string.digits + string.ascii_uppercase
# characters = '我是中国人'
# print(characters)
#
# width, height, n_len, n_class = 170, 80, 4, len(characters) + 1
#
# generator = ImageCaptcha(width=width, height=height)
# random_str = ''.join([random.choice(characters) for j in range(4)])
# img = generator.generate_image(random_str)
# plt.imshow(img)
# plt.title(random_str)
# pylab.show()

# import os
# import pygame
# import random
# import matplotlib.pyplot as plt
# import pylab
#
# pygame.init()
#
# text = ['我是中国人', '是', '中', '国', '人']
#
# font = pygame.font.Font(os.path.join('fonts', 'C:\\Windows\\Fonts\\simkai.ttf'), 60)
#
# for word in text:
#     rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
#     pygame.image.save(rtext, '{0}.jpg'.format(word))

# scene_validation_annotations_20170908 = 'E:\\LearningDeepData\\ai_challenger_scene_validation_20170908\\scene_validation_annotations_20170908.json'
# ai_challenger_scene_validation_20170908 = 'E:\\LearningDeepData\\ai_challenger_scene_validation_20170908\\scene_validation_images_20170908\\{0}'
#
# validation_json_file = open(scene_validation_annotations_20170908)
# validation_json_data = json.load(validation_json_file)
# validation_json_file.close()
#
# cc = {}
#
# for i in validation_json_data:
#     image_id = i['image_id']
#     label_id = str(i['label_id'])
#     cc[label_id] = cc.get(label_id, '') + ',' + image_id
#
# dd = []
# for key, value in cc.items():
#     image_ids = random.sample(value.split(',')[1:], 5)
#     for image_id in image_ids:
#         item = {}
#         item['label_id'] = str(key)
#         item['image_id'] = image_id
#         dd.append(item)
# print_list_info(dd)


# print(label_map)
# print(label_map_converse)
#
# for i in range(80):
#     print(str(i) + ':' + str(i) + ',')
