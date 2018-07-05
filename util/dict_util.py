#!/bin/bash
# -*- coding=utf-8 -*-
import config as con
from util.pickle_util import *


class DictUtil():
    def __init__(self):
        pass

    def count_value(self, source_dict):
        result = {}
        for key, value in source_dict.items():
            source_dict[key] = source_dict.get(key, 0) + 1
        return result


if __name__ == '__main__':
    dictUtil = DictUtil()
    pickleUtil = PickleUtil()
    imageId_labelId_dict = pickleUtil.pickle_load(con.pick['imageId_labelId_53879_dict'])

    with open('result.txt', 'w') as f:
        for key, value in imageId_labelId_dict.items():
            f.write(key + '\t' + value + '\n')
