#!/bin/bash
# -*-coding=utf-8-*-
import re
import os
import random
from PIL import Image
import numpy as np
from util.file_util import *
from util.label_util import *
import config as con
import util.log_util
import pickle
from util.file_util import *
from numpy import random

'''
@FileName: matrix.py
@Author：liangzhicheng tanzhiyuan
@Create date:  2017-07-15
@description：图片转矩阵,输出mat文件用于存放矩阵信息
@File URL: https://github.com/liangzhicheng120/contest
'''


class ImageUtil():
    def test(self, image_name):
        image_matrix = self.image_to_matrix(image_name)
        image_matrix_normalization = self.normalization(image_matrix)
        print(np.amin(image_matrix_normalization))  # 最大值
        print(np.amax(image_matrix_normalization))  # 最小值

    def normalization(self, image_matrix):
        '''
        图片归一化
        :param image_gray:图片
        :return: 矩阵
        '''
        image_matrix = ((image_matrix - 128) / 128.0)
        # image_shape = [3, self._image_height, self._image_width]
        # image_matrix = np.reshape(image_matrix, image_shape)
        # image_matrix = 1 + np.floor(image_matrix - image_matrix.mean())
        return image_matrix

    def image_to_matrix(self, image_name):
        '''
        图片转成矩阵
        :return:
        '''
        image_handle = Image.open(image_name)
        image = image_handle.resize((con.image['image_width'], con.image['image_height']), Image.ANTIALIAS)
        image_matrix = np.array(image)
        image_matrix_normalization = image_matrix
        # print(image_name, np.shape(image_matrix_normalization))
        # image_matrix_normalization = self.enhance_data(image_matrix_normalization)
        # image_matrix_normalization = self.normalization(image_matrix)
        return image_matrix_normalization

    def image_to_matrix_16(self, image_name):
        '''
        图片转成矩阵
        :return:
        '''
        image_handle = Image.open(image_name)
        image = image_handle.resize((con.image['image_width'], con.image['image_height']), Image.ANTIALIAS)
        image_matrix = np.array(image)
        image_matrix_normalization = []

        # 镜像
        # image_matrix_normalization = self.enhance_data_mirror(image_matrix)

        # 对比度
        # image_matrix_normalization = self.enhance_data_contrast(image_matrix)

        # 随机切割16张
        for i in range(16):
            image_matrix_normalization.append(self.enhance_data(image_matrix))

        # image_matrix_normalization = self.normalization(image_matrix)
        return image_matrix_normalization

    def enhance_data(self, image_matrix):
        '''
        数据增强
        图像切片
        :return:
        '''
        section_width_point = random.randint(0, con.image['image_width'] - con.image['width'] - 1)
        section_height_point = random.randint(0, con.image['image_height'] - con.image['height'] - 1)
        reimage_matrix = image_matrix[section_width_point:section_width_point + con.image['width'],
                         section_height_point:section_height_point + con.image['height'], 0:con.image['dimension']]
        return reimage_matrix

    def enhance_data_mirror(self, image_matrix):
        '''
        镜像
        :param image_matrix:
        :return:
        '''
        image_matrix_mirror = np.zeros([con.image['width'], con.image['height'], con.image['dimension']])
        for i in range(con.image['width']):
            image_matrix_mirror[:, con.image['height'] - 1 - i, :] = image_matrix[:, i, :]
        return image_matrix_mirror

    def enhance_data_contrast(self, image_matrix):
        '''
        翻转
        :param image_matrix:
        :return:
        '''
        contrast = 0.5
        return contrast * image_matrix

    def matrix_to_image(self, matrix, image_name):
        '''
        矩阵转图片 供测试用
        :param matrix:
        :return:
        '''
        r = Image.fromarray(matrix[:, :, 0]).convert('L')
        g = Image.fromarray(matrix[:, :, 1]).convert('L')
        b = Image.fromarray(matrix[:, :, 2]).convert('L')
        image_rgb = Image.merge("RGB", (r, g, b))
        # image_r = Image.merge("RGB", (r, r, r))
        image_rgb.save('{0}.jpg'.format(image_name), 'png')
        # image_r.save(self._reImageName + "_r.png", 'png')b

    def mean(source_list):
        result = 0
        for i in source_list:
            result = sum(i) / len(i)
        return result

    def splist(source_list, splist_size):
        '''
        分隔列表
        :param batch_size:
        :return:
        '''
        source_list = list(source_list)
        result_list = [source_list[i:i + splist_size] for i in range(len(source_list)) if i % splist_size == 0]
        return result_list, len(result_list)


if __name__ == '__main__':
    imageUtil = ImageUtil()
    fileUtil = FileUtil()
    # for i in range(37, 38):
    #     files, _ = fileUtil.get_files('E:\\LearningDeepOther\\{0}'.format(i))
    #     print(len(files))
    #     for name in files:
    #         array = imageUtil.image_to_matrix_16('E:\\LearningDeepOther\\{0}\\'.format(i) + name)
    #         for j in range(len(array)):
    #             imageUtil.matrix_to_image(array[j], 'E:\\LearningDeepBatch\\80_16_1\\' + name.split('.')[
    #                 0] + '_' + str(j))
    #
    # array = imageUtil.image_to_matrix_16('gas_station_22.jpg')
    # imageUtil.matrix_to_image(array, 'test')
    source_path = 'E:\\LearningDeepBatch\\80_2_4\\'
    target_path = 'E:\\LearningDeepBatch\\80_2_16\\'
    files, files_len = fileUtil.get_files(source_path)
    for name in files:
        matrix = imageUtil.image_to_matrix_16(source_path + name)
        # imageUtil.matrix_to_image(matrix, target_path + name.split('.')[0] + '_5')
        for j in range(len(matrix)):
            imageUtil.matrix_to_image(matrix[j], target_path + name.split('.')[0] + '_' + str(j))
    print('end')
