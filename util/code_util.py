#!/bin/bash
# -*-coding:utf-8 -*-
import re
import os
import sys
import linecache
import codecs
import numpy as np
from util.base_util import *
import config as con
from util.label_util import *
from util.string_util import *

'''
# str_to_matrix 字符串转二进制矩阵
# matrix_to_str 矩阵转字符串
# test 测试
'''


class CodeUtil(BaseUtil):
    def str_to_matrix(self, str):
        '''
        字符串转二进制矩阵,每行代表一个字符
        :param str: 字符串
        :return:
        '''
        result = map(lambda x: list(self.add_zero(self.str_to_bin(x))), list(str.strip()))  # 使用补零
        result = map(lambda x: list(map(eval, x)), result)
        result = np.array(list(result))
        return result

    def int_list(self, source_list):
        '''
        数值型列表转整型列表
        :param source_list:
        :return:
        '''
        return list(map(int, source_list))

    def matrix_to_str(self, matrix):
        '''
        矩阵转字符串
        :param matrix:矩阵
        :return:
        '''
        result = matrix.tolist()
        result = list(map(self.int_list, result))
        result = list(map(lambda x: ''.join(list(map(str, x))), result))
        result = map(self.bin_to_str, result)
        result = list(result)
        result = ''.join(result)
        return result

    def str_to_format_matrix(self, str, format_char=con.char['format_char'], max_len=con.char['max_len']):
        '''
        字符串转规则矩阵,矩阵大小由最大字符串决定
        :param str:
        :param format_char:
        :param max_len:
        :return:
        '''
        add_len = max_len - len(str)
        add_cahr = ''.join(list(map(lambda x: format_char, range(add_len))))
        format_matrix = self.str_to_matrix(str + add_cahr)
        return format_matrix

    def num_to_format_matrix(self, num):

        pass

    def test(self, str):
        matrix = self.str_to_format_matrix(str)
        str = self.matrix_to_str(matrix)
        print('======================\n{0}\n{1}'.format(matrix, str))




# if __name__ == '__main__':
#     codeUtil = CodeUtil()
#     labelUtil = LabelUtil()
#     stringUtil = StringUtil()
#     _step = 0
#     max_len = 0
#     imageId_caption_dict = labelUtil.pick_load(con.pick['imageId_caption_dict'])
#     for imageId, captions in imageId_caption_dict.items():
#         if _step % 100 == 0:
#             print('_step:{0}'.format(_step))
#         for caption in captions:
#             actual = stringUtil.replace_to_black(caption, con.char['replace_char'])
#             change = codeUtil.matrix_to_str(codeUtil.str_to_matrix(actual))
#             max_len = len(actual) if len(actual) > max_len else max_len
#             max_len = (len(actual) > max_len) * len(actual) + (len(actual) <= max_len)*max_len
#             if actual != change:
#                 print('imageId:{0} sentence:{1}'.format(imageId, actual))
#         _step += 1
#         print('max_len:{0}'.format(max_len))
#
# codeUtil = CodeUtil()
# codeUtil.test('我是中国人')  # 简体字
# codeUtil.test('123456')  # 数字
# codeUtil.test('!@#$%^&*()')  # 英文特殊字符
# codeUtil.test('！@#￥%……&*（）')  # 中文特殊字符
# codeUtil.test('犇猋骉麤毳淼掱焱垚赑')  # 生僻字
# codeUtil.test('this is a test')
# codeUtil.test('にほんご、にっぽんご')  # 日语
# codeUtil.test('???')  # 韩语  暂不支持
# matrix = codeUtil.str_to_format_matrix('我是中国人')
# print(matrix)
