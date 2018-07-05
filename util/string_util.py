#!/bin/bash
# -*- coding:utf-8 -*-
from util.log_util import *


class StringUtil(object):
    def __init__(self):
        pass

    def replace_to_black(self, str, content):
        '''
        替代字符为空格
        :param str:
        :param content:
        :return:
        '''
        for char in content:
            str = str.replace(char, '')
        return str
