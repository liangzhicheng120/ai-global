#!/bin/bash
# -*-coding=utf-8-*-
import pickle
import config as con
from util.log_util import *
import json


class PickleUtil():
    def __init__(self):
        self.dictionary_list = []
        self.dictionary_index = {}
        self.dictionary_word = {}
        self.chinese_pickle_name = 'chinese_dictionary_{0}_{1}.pik'
        self.chinese_word_pickle_name = 'chinese_word_dictionary_{0}_{1}.pik'
        self.imageId_caption_dict_name = 'imageId_caption_dict.pik'
        pass

    def pickle_dump(self, object, file_name):
        pickle.dump(object, open(file_name, 'wb'), protocol=2)

    def pickle_load(self, file_name):
        print('pickle_load:{0}'.format(file_name))
        return pickle.load(open(file_name, 'rb'))

    @print_func_done
    def chinese_pickle(self, file_name):
        '''
        构造汉字Id
        :param file_name:
        :return:
        '''
        with open(file_name, 'r', encoding=con.char['encoding']) as f:
            for line in f:
                self.dictionary_list += list(line.strip())
        self.dictionary_list = list(set(self.dictionary_list))
        for index, word in enumerate(self.dictionary_list):
            self.dictionary_index[index] = word
            self.dictionary_word[word] = index
        list_len = len(self.dictionary_list)
        self.pickle_dump(self.dictionary_index, self.chinese_pickle_name.format(list_len, 'index'))
        self.pickle_dump(self.dictionary_word, self.chinese_pickle_name.format(list_len, 'word'))
        print_dict_info(self.pickle_load(self.chinese_pickle_name.format(list_len, 'index')), is_print_all=False)
        print_dict_info(self.pickle_load(self.chinese_pickle_name.format(list_len, 'word')), is_print_all=False)

    @print_func_done
    def imageId_capation_pickle(self, file_name):
        '''
        构造图片Id和描述
        :param file_name:
        :return:
        '''
        result = {}
        with open(file_name, 'r', encoding=con.char['encoding']) as input:
            labels = json.load(input)
            for i in range(len(labels)):
                result[labels[i]['image_id']] = labels[i]['caption']
        self.pickle_dump(result, self.imageId_caption_dict_name)
        print_dict_info(self.pickle_load(self.imageId_caption_dict_name), is_print_all=False)

    def chinese_word_pickle(self, file_name):
        '''
        构造汉字词语Id
        :param file_name:
        :return:
        '''
        with open(file_name, 'r', encoding=con.char['encoding']) as f:
            for line in f:
                words = line.strip().split(' ')
                for word in words:
                    self.dictionary_list.append(word)
        self.dictionary_list = list(set(self.dictionary_list))
        for index, word in enumerate(self.dictionary_list):
            self.dictionary_index[index] = word
            self.dictionary_word[word] = index
        list_len = len(self.dictionary_list)
        self.pickle_dump(self.dictionary_index, self.chinese_word_pickle_name.format(list_len, 'index'))
        self.pickle_dump(self.dictionary_word, self.chinese_word_pickle_name.format(list_len, 'word'))
        print_dict_info(self.pickle_load(self.chinese_word_pickle_name.format(list_len, 'index')), is_print_all=False)
        print_dict_info(self.pickle_load(self.chinese_word_pickle_name.format(list_len, 'word')), is_print_all=False)


# if __name__ == '__main__':
#     pickleUtil = PickleUtil()
    # pickleUtil.chinese_pickle(con.label['训练汉字'])
    # pickleUtil.imageId_capation_pickle(con.json['caption_train_annotations'])
    # pickleUtil.chinese_word_pickle(con.w2v['w2v_train_file'])

    # max = 0
    # sent = ''
    # with open(con.w2v['w2v_train_file'], 'r', encoding='utf-8') as f, open('result.txt', 'w', encoding='utf-8') as r:
    #     for line in f:
    #         sentences = line.strip().split('。')
    #         for sentence in sentences:
    #             _len = len(sentence.strip().split(' '))
    #             max = max if max > _len else  _len
    #             r.write('{0}\t{1}\n'.format(_len, sentence))
    #             # print(_len,sentence)
    # print(max)
