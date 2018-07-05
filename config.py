#!/bin/bash
# -*- coding:utf-8 -*-
import sys

#### 根路径 ####
data_dir = 'E:\\LearningDeepData\\'

### 模型文件属性 ###
model = {
    'w2v_size200': 'w2v_size200.model',
    'size': 200,
}

### 图片文件属性 ###
image = {
    '1w': '1w',
    'all': 'ai_challenger_caption_train_20170902\\caption_train_images_20170902\\',
    'image_height': 128,  # 180
    'image_width': 128,  # 180
    'height': 128,
    'width': 128,
    'dimension': 3,
    'scene_train': 'ai_challenger_scene_train_20170904\\scene_train_images_20170904',
    'scene_test': 'ai_challenger_scene_test_a_20170922\\scene_test_a_images_20170922',
}

### 标签文件属性 ###
label = {
    '中国汉字大全': '中国汉字大全.txt',
    '训练汉字': '训练汉字.txt',
}

### json文件属性 ###
json = {
    'caption_train_annotations': 'caption_train_annotations.json',
    'scene_train_annotations': 'ai_challenger_scene_train_20170904\\scene_train_annotations_20170904.json',
}
### 图片ID、描述腌制文件属性 ###
pick = {
    'imageId_caption_dict': 'imageId_caption_dict.pik',
    'chinese_dictionary_20963_index': 'chinese_dictionary_20963_index.pik',
    'chinese_dictionary_20963_word': 'chinese_dictionary_20963_word.pik',
    'sceneId_classes_dict': 'sceneId_classes_dict.pik',
    'imageId_labelId_53879_dict': 'imageId_labelId_53879_dict.pik',
}

### w2v文件属性 ###
w2v = {
    'w2v_train_file': 'w2v_train_file.txt',
}

### 字符属性 ###
char = {
    'replace_char': [' ', '\n', '\u3000', '\t'],
    'max_len': 50,
    'format_char': '.',
    'bit_size': 16,
    'encoding': 'utf-8'
}


def add_dir(dir, dict_list):
    for dict in dict_list:
        for k in dict.keys():
            if type(dict[k]) != type(0) and dict[k] != '.':
                dict[k] = dir + dict[k]


add_dir(dir=data_dir, dict_list=[model, image, label, json, pick, w2v])

label_map = {'0': '2',
             '1': '4',
             '2': '14',
             '3': '18',
             '4': '21',
             '5': '22',
             '6': '23',
             '7': '24',
             '8': '48',
             '9': '53',
             '10': '56',
             '11': '61',
             '12': '63',
             '13': '68',
             '14': '74',
             '15': '1',
             '16': '8',
             '17': '9',
             '18': '10',
             '19': '11',
             '20': '12',
             '21': '13',
             '22': '15',
             '23': '17',
             '24': '19',
             '25': '20',
             '26': '25',
             '27': '27',
             '28': '28',
             '29': '29',
             '30': '30',
             '31': '31',
             '32': '33',
             '33': '34',
             '34': '35',
             '35': '37',
             '36': '39',
             '37': '40',
             '38': '41',
             '39': '43',
             '40': '44',
             '41': '45',
             '42': '47',
             '43': '51',
             '44': '52',
             '45': '54',
             '46': '55',
             '47': '58',
             '48': '59',
             '49': '62',
             '50': '64',
             '51': '66',
             '52': '69',
             '53': '70',
             '54': '73',
             '55': '75',
             '56': '77',
             '57': '78',
             '58': '0',
             '59': '3',
             '60': '5',
             '61': '6',
             '62': '7',
             '63': '16',
             '64': '26',
             '65': '32',
             '66': '36',
             '67': '38',
             '68': '42',
             '69': '46',
             '70': '49',
             '71': '50',
             '72': '57',
             '73': '60',
             '74': '65',
             '75': '67',
             '76': '71',
             '77': '72',
             '78': '76',
             '79': '79'}
label_map_converse = {v: k for k, v in label_map.items()}
