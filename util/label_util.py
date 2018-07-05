#!/bin/bash
# -*- coding:utf-8 -*-
import config as con
import json
import pickle


class LabelUtil():
    def __init__(self):
        pass

    def get_capation(self, key, dict):
        '''
        根据imageId返回描述
        :param key:
        :param dict:
        :return:
        '''
        return dict.get(key.strip())

    def splist(self, source_list, splist_size):
        '''
        分隔列表
        :param batch_size:
        :return:
        '''
        source_list = list(source_list)
        result_list = [source_list[i:i + splist_size] for i in range(len(source_list)) if i % splist_size == 0]
        return result_list, len(result_list)

    def id_to_classes(self, id_dict, classes_list):
        '''
        id classes 映射
        :param id_dict:
        :param classes_list:
        :return:
        '''
        result = list(map(lambda x: id_dict.get(str(x)), classes_list))
        return result
