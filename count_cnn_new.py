#!/bin/bash
# -*- coding:utf-8 -*-
from util.log_util import *

count_dict = {}
act_dict = {}
act_num = 0
tot_num = 0
with open('count.txt', 'r') as f:
    for line in f:
        index, act, pre = line.strip().split('\t')
        tot_num += 1
        pre = pre.split(',')
        act_dict[int(act)] = act_dict.get(int(act), 0) + 1
        if act in pre:
            act = int(act)
            count_dict[act] = count_dict.get(act, 0) + 1
            act_num += 1
            # print(act_num)
            # print('acc:{0}'.format(act_num / tot_num))

print_dict_info(count_dict)
