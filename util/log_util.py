#!/bin/bash
# -*- coding:utf-8 -*-
import time
from functools import wraps


def print_func_time(function):
    '''
    计算程序运行时间
    :param function:
    :return:
    '''

    @wraps(function)
    def func_time(*args, **kwargs):
        t0 = time.clock()
        result = function(*args, **kwargs)
        t1 = time.clock()
        print("Total running time: %s s" % (str(t1 - t0)))
        return result

    return func_time


def print_func_done(function):
    '''
    打印信息
    :param function:
    :param info:
    :return:
    '''

    @wraps(function)
    def func_info(*args, **kwargs):
        result = function(*args, **kwargs)
        print('=====>', function.__name__, 'is done')
        return result

    return func_info


def print_list_info(source_list, is_print_all=True):
    '''
    打印列表信息
    :param list:
    :return:
    '''
    for line in source_list:
        if is_print_all:
            print(line)
        else:
            print(line)
            break
    print('<size:{0}>'.format(len(source_list)))
    print(source_list.__class__)
    print()


def print_dict_info(source_dict, is_print_all=True):
    '''
    打印字典信息
    :param source_dict:
    :return:
    '''
    count = len(source_dict)
    for key, value in source_dict.items():
        if is_print_all:
            print(key, value)
        else:
            print(key, value)
            break
    print('<size:{0}>'.format(count))
    print(source_dict.__class__)
    print()
