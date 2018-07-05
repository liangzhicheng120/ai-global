#!/bin/bash
# -*- coding:utf-8 -*-

from w2v.word2vec import *
import config as con


class Word2VecTest(Word2Vec):
    def __init__(self):
        pass

    def test_w2v_model(self, train_model, train_corpus):
        '''
        测试w2v模型
        :param train_corpus: 语料 [id \t sentence_1 \t sentence_2]
        :return:
        '''
        self.load_W2V_Model(train_model)
        result = []
        with open(train_corpus, 'r', encoding=con.char['encoding']) as f:
            for line in f:
                id, sentence_1, sentence_2 = line.strip().split('\t')
                distence = self.compute_distance(sentence_1, sentence_2)
                result.append(id + '\t' + sentence_1 + '\t' + sentence_2 + '\t' + str(distence) + '\n')
        self.write_2_file('test_w2v_model_{0}.xlsx'.format(len(result)), result)
        pass


if __name__ == '__main__':
    word2VecTest = Word2VecTest()
    word2VecTest.test_w2v_model(train_model=con.model['w2v_size200'], train_corpus=con.label['label-test'])
    pass
