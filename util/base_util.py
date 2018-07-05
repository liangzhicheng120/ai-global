#!/bin/bash
# -*-coding:utf-8 -*-
import re
import os
import sys
import linecache
import codecs
import numpy as np
import config as con

'''
# bin_to_str 二进制转字符串
# bin_to_dec 二进制转十进制
# str_to_bin 字符串转二进制
# read_file 读取文件,统一转换成unicode编码
# add_zero 补零
'''


class BaseUtil(object):
    def bin_to_str(self, str):
        '''
        二进制转字符串
        :param str: 二进制数字型字符串
        :return:
        '''
        return chr(self.bin_to_dec(str))

    def bin_to_dec(self, str):
        '''
        二进制转十进制
        :param str:二进制数字型字符串
        :return:
        '''
        return int(str, 2);

    def str_to_bin(self, str):
        '''
        字符串转二进制
        :param str: 任意字符串
        :return: 二进制编码
        '''
        return '{0:b}'.format((ord(str)))

    def read_file(self, fileName):
        '''
        读取文件,统一转换成unicode编码
        :param fileName:文件名
        :return:
        '''
        f = codecs.open(fileName, encoding='utf-8')
        try:
            f.read(1)
        except:
            f = codecs.open(fileName, encoding='gbk')
        return f

    def add_zero(self, str, bit_size=con.char['bit_size']):
        '''
        补零
        英文1字节,汉字3字节,生僻字4-6字节,默认3字节
        :param str: 字符串
        :param num: 数字
        :return:
        '''
        return str.zfill(bit_size)

    def num_to_bin(self, num):
        '''
        十进制转二进制
        :param num:
        :return:
        '''
        return self.add_zero(bin(num).replace('0b', ''), bit_size=12)
