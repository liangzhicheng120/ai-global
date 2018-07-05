#!/bin/bash
# -*- coding:utf-8 -*-

import config as con
from util.file_util import *
from w2v.word2vec import *


class DisposeLable(Word2Vec):
    def __init__(self):
        pass

    def json_2_file(self, in_fileName, out_fileName, ex_head, head):
        '''
        json 文件转 file文件
        :param in_fileName:
        :param out_fileName:
        :param ex_head:
        :param head:
        :return:
        '''
        result = []
        with open(in_fileName, 'r') as input:
            result.append('\t'.join(ex_head) + '\t' + '\t'.join(head) + '\n')
            labels = json.load(input)
            labels_len = len(labels)
            for i in range(labels_len):
                result.append('{0}\t{1}\t{2}\t{3}\n'.format(str(i + 1), labels[i][head[0]], labels[i][head[1]], labels[i][head[2]]))
        self.write_2_file(out_fileName, result)

    def select_label(self, img_file, in_fileName, out_fileName, ex_size):
        '''
        挑选指定图片的label
        :param img_file:
        :param in_fileName:
        :param out_fileName:
        :param ex_size:
        :return:
        '''
        result = []
        name_list = self.get_fileName(img_file, 'jpg', ex_size=ex_size)
        pattern = '|'.join(name_list)
        for line in self.read_file(in_fileName):
            _name = line.split('\t')[-1]
            if re.match(r'{0}'.format(pattern), _name):
                result.append(line + '\n')
        self.write_2_file(out_fileName, result)

    def split_all_label(self, in_fileName, out_fileName):
        '''
        分隔label文件
        :param in_fileName:
        :param out_fileName:
        :return:
        '''
        result = []
        with open(in_fileName, encoding=con.char['encoding']) as f:
            for line in f:
                index, url, caption, image_id = line.strip().split('\t')
                caption = self.replace_2_black(caption, ["'", "[", "]"])
                result.append(index + '\t' + url + '\t' + '\t'.join(caption.split(',')) + '\t' + image_id + '\n')
        self.write_2_file(out_fileName, result)

    def jieba_cut_label(self, in_fileName):
        '''
        切词
        :param in_fileName:
        :return:
        '''
        result = []
        with open(in_fileName, 'r', encoding=con.char['encoding']) as input:
            labels = json.load(input)
            for i in range(len(labels)):
                for line in labels[i]['caption']:
                    result.append(' '.join(self.jieba_cut(line)) + ' 。 ')
        self.write_2_file('w2v_train_file.txt', result)
        pass

# if __name__ == '__main__':
#     disposeLable = DisposeLable()
#     # disposeLable.json_2_file(con.json['caption_train_annotations'], 'label-all.txt', ['index'], ['url', 'caption', 'image_id'])
#     # disposeLable.select_label(con.image['1w'], con.label['label-all'], 'label-1w.txt', ex_size=10000)
#     # disposeLable.split_all_label(con.label['label-all'], 'label-all-split.txt')
#     disposeLable.jieba_cut_label(con.json['caption_train_annotations'])
#     pass
