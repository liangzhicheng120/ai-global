#!/bin/bash
# -*- coding=utf-8 -*-
import os
import re
import config as con
import csv
import json
import sys
from util.string_util import *
from util.log_util import *


class FileUtil(StringUtil):
    def __init__(self):
        pass

    def get_file_name(self, file_dir, file_suf, ex_size):
        '''
        获取指定路径下指定后缀的的文件名
        :param fileDir:
        :param fileSuf:
        :return:
        '''
        result = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.{0}'.format(file_suf):
                    result.append(os.path.join(file))
        _len = len(result)
        if ex_size != _len:
            raise Exception('指定路径下文件个数不符合预期数量 ex:{0} fact:{1}'.format(ex_size, _len))
        return result

    def read_file(self, fileName):
        '''
        去除制表符读取文件
        :param fileName:
        :return:
        '''
        result = []
        with open(fileName, 'r', encoding=con.char['encoding']) as f:
            for line in f:
                result.append(line.strip())
        return result

    @print_func_done
    def write_2_file(self, fileName, list):
        '''
        将列表内容写到文件中
        :param fileName:
        :param list:
        :return:
        '''
        with open(sys.path[1] + '\\' + fileName, 'w', encoding=con.char['encoding']) as f:
            f.writelines(list)

    def get_files(self, file_dir, file_suf='jpg'):
        '''
        获取指定路径下指定后缀的的文件名
        :param fileDir:
        :param fileSuf:
        :return:
        '''
        result = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.{0}'.format(file_suf):
                    result.append(os.path.join(file))
        return result, len(result)
