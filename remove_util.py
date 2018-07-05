#!/bin/bash
# -*- coding=utf-8 -*-
import config as con
from util.pickle_util import *
from util.log_util import *
import json
import shutil
import os
from util.file_util import *
import random
import json

## 初始化工具类
fileUtil = FileUtil()


## 根据名字复制图片

def copy_image(classes_id, classes):
    # 类别
    # classes_id, classes = 0, 'airport_terminal'
    # 旧图片位置
    old_image_dir = 'E:\\LearningDeepData\\ai_challenger_scene_train_20170904\\scene_train_images_20170904\\'
    # 新图片位置
    new_image_dir = 'E:\\LeariningDeepOther\\{0}\\'.format(classes_id)
    # 计数
    count = 0
    # 创建路径
    os.makedirs(new_image_dir)
    with open(con.json['scene_train_annotations'], 'r') as f:
        scene_train_annotations = json.load(f)
        for i in range(len(scene_train_annotations)):
            if scene_train_annotations[i]['label_id'] == str(classes_id):
                image_name = scene_train_annotations[i]['image_id']
                new_image_name = '{0}_{1}.jpg'.format(classes, count)
                shutil.copy(old_image_dir + image_name, new_image_dir + new_image_name)
                count += 1
    print(count)


# 分类
# with open('E:\\LearningDeepData\\scene_classes.txt', 'r') as f:
#     for line in f:
#         classes_id, classes = line.strip().split('\t')
#         classes_id = classes_id.decode('utf-8').encode('gbk', 'ignore')
#         classes = classes.replace('/', '_')
#         copy_image(classes_id, classes)

# 补充图片
# def add_image():
#     # 总图片数
#     total_len = 1000
#     # 补充图片
#     other_classes, other_classes_len = fileUtil.get_files('E:\\LearningDeepData\\ai_challenger_scene_other\\0\\', file_suf='jpg')
#     # 原来图片
#     source_classes, source_classes_len = fileUtil.get_files('E:\\LeariningDeepOther\\0\\', file_suf='jpg')
#     # 需要图片数
#     need_len = total_len - source_classes_len
#     # 计步
#     step = source_classes_len
#
#     old_image_dir = 'E:\\LearningDeepData\\ai_challenger_scene_other\\0\\'
#     new_image_dir = 'E:\\LeariningDeepOther\\0\\'
#
#     for image_name in other_classes[:need_len]:
#         new_image_name = '_'.join(image_name.split('_')[:2]) + '_' + str(step) + '.jpg'
#         shutil.copy(old_image_dir + image_name, new_image_dir + new_image_name)
#         step += 1


# 构造
def other():
    # 当前类别
    current_class = 0
    num = 0
    copy_size = 21
    for class_id in range(80):
        if class_id == 0:
            continue
        old_image_dir = 'E:\\LearningDeepOther\\{0}\\'.format(class_id)
        new_image_dir = 'E:\\LearningDeepOther\\{0}\\'.format(current_class)
        files_name, _ = fileUtil.get_files(old_image_dir, 'jpg')
        files_name_rand = random.sample(files_name, copy_size)
        for image_name in files_name_rand:
            new_image_name = 'other_{0}.jpg'.format(num)
            shutil.copy(old_image_dir + image_name, new_image_dir + new_image_name)
            num += 1


# 构造json文件
def make_json():
    result = []
    count = 0
    class_id = 0
    files_name, _ = fileUtil.get_files('E:\\LearningDeepOther\\{0}\\'.format(class_id))
    for image_name in random.sample(files_name, len(files_name)):
        current = {}
        if 'other' in image_name:
            current['label_id'] = '0'
            current['image_id'] = image_name
            result.append(current)
        else:
            current['label_id'] = '1'
            current['image_id'] = image_name
            result.append(current)
    json.dump(result, open('0_image_lable.json', 'w'))


def make_validation_json():
    classes = 0
    result = []
    json_file = 'E:\\LearningDeepData\\ai_challenger_scene_validation_20170908\\scene_validation_annotations_20170908.json'
    json_data = json.load(open(json_file, 'r'))
    # print(len(json_data))  # 7120
    count = 0
    other = 0
    for i in range(len(json_data)):
        current = {}
        if str(classes) == json_data[i]['label_id']:
            if count >= 50:
                continue
            count += 1
            current['label_id'] = '1'
            current['image_id'] = json_data[i]['image_id']
            result.append(current)
        else:
            if other >= 50:
                continue
            current['label_id'] = '0'
            current['image_id'] = json_data[i]['image_id']
            result.append(current)
            other += 1
    # print(str(classes), count)
    # # 随机打乱
    # result = random.sample(result, len(result))
    # # 输出
    json.dump(result, open('{0}_image_lable_validation.json'.format(classes), 'w'))
    print_list_info(result)


def make_json():
    result = []
    files, files_len = fileUtil.get_files('E:\\gas')
    with open(con.json['scene_train_annotations'], 'r') as f:
        json_data = json.load(f)
        for i in range(len(json_data)):
            item = {}
            label_id, image_id = json_data[i]['label_id'], json_data[i]['image_id']
            item['image_id'] = image_id
            item['label_id'] = label_id
            result.append(item)
    for image_id in files:
        item = {}
        item['image_id'] = image_id
        item['label_id'] = '55'
        result.append(item)
    result = random.sample(result, len(result))
    json.dump(result, open('scene_train_annotations_{0}.json'.format(54355), 'w'))


def rename():
    for i in [63, 67, 69, 70, 72, 74, 76, 77]:
        source_path = 'E:\\LearningDeepBatch\\{0}\\'.format(i)
        target_parh = 'E:\\LearningDeepBatch\\{0}\\'.format(i)
        files, lens = fileUtil.get_files(source_path)
        add = 400 - lens
        print(len(files[:add]))
        for num, name in enumerate(files[:add]):
            shutil.copy(source_path + name, target_parh + '{0}_{1}.jpg'.format(files[0].split('.')[0], num))
