#!/bin/bash
# -*-coding=utf-8-*-
import numpy as np


class NumpyUtil():
    def __init__(self):
        pass

    def count_array(arr, size):
        '''
        计算二维矩阵元素最多的前n项
        :param size:
        :return:
        '''
        result = np.array(sorted([[np.sum(arr == i), i] for i in set(arr.flat)]))
        return result[-size:, 1]
