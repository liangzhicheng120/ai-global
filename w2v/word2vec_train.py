#!/bin/bash
# -*- coding:utf-8 -*-
import gensim, logging
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
from w2v.word2vec import *
import config as con


class Word2VecTrain(Word2Vec):
    def __init__(self):
        pass

    def train_w2v_model(self, in_fileName, size=200, window=6, min_count=1):
        '''
        获取w2v训练模型
        :param in_fileName:
        :param size:
        :param window:
        :param min_count:
        :return:
        '''
        sentence = word2vec.Text8Corpus(in_fileName)
        model = word2vec.Word2Vec(sentence, size=size, window=window, min_count=min_count);
        model.save(sys.path[1] + '\\w2v_size{0}.model'.format(size))
        pass


if __name__ == '__main__':
    word2VecTrain = Word2VecTrain()
    word2VecTrain.train_w2v_model(con.w2v['w2v_train_file'], size=200, window=6, min_count=1)
    pass
