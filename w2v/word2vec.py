#!/bin/bash
# -*- coding:utf-8 -*-
from __future__ import division
import gensim
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import jieba
import jieba.posseg as pseg
import config as con
import sys
from util.file_util import *


class Word2Vec(FileUtil):
    def __del__(self):
        print("----delete object----")

    def load_W2V_Model(self, ModelName):
        '''
        load word2vec model
        '''
        self.model = gensim.models.word2vec.Word2Vec.load(ModelName)
        print('loadModel:{0}'.format(ModelName))

    def word_2_vec(self, words):
        '''
        词转向量
        :param words:
        :return:
        '''
        Vec = []
        num = len(words)
        for word in words:
            vec = self.model[word]
            Vec.append(vec)
            # print(vec)
        Vec = sum(Vec) / num
        return Vec

    def sentence_2_vec(self, sentences):
        '''
        句子转向量
        :param sentences:
        :return:
        '''
        Vec = []
        for sentence in sentences:
            words = self.jieba_cut(self.replace_to_black(sentence.strip(), con.char['replace_char']))
            vec = self.word_2_vec(words)
            Vec.append(vec)
        return Vec

    def train(self, X_train, Y_train):
        '''
        训练模型
        :param X_train:
        :param Y_train:
        :return:
        '''
        Vec = self.s2v(X_train)
        self.clf = GaussianNB()
        self.clf.fit(Vec, Y_train)

    def predict(self, Test):
        '''
        预测模型
        :param Test:
        :return:
        '''
        Vec = self.s2v(Test)
        result = self.clf.predict(Vec)
        print(result)

    def jieba_cut(self, sentence):
        '''
        切词
        :param sentence:
        :return:
        '''
        words_flag = pseg.cut(sentence)
        words = []
        for word, flag in words_flag:
            words.append(word)
        return words

    def save_NBmodel(self, model_name):
        '''
        保存模型
        :param model_name:
        :return:
        '''
        joblib.dump(self.clf, model_name)
        print("Save Seccessfully")

    def load_NBmodel(self, model_name, word2vecModel=con.model['w2v_size200']):
        '''
        加载模型
        :param model_name:
        :param word2vecModel:
        :return:
        '''
        self.load_W2V_Model(word2vecModel)
        self.clf = joblib.load(model_name)

    def compute_distance(self, sentence_1, sentence_2):
        '''
        计算两个句子间的距离
        :param sentence_1:
        :param sentence_2:
        :return:
        '''
        vec = self.sentence_2_vec([sentence_1, sentence_2])
        R = sum((vec[0] - vec[1]) ** 2) ** 0.5
        return R

    def get_w2v_model_matrix(self, str, w2v_model):
        '''
        获取w2v模型中的matrix值
        :param str:
        :param w2v_model:
        :return:
        '''
        model = gensim.models.word2vec.Word2Vec.load(w2v_model)
        matrix = model[str]
        return matrix


# if __name__ == "__main__":
    #     X_train = np.array([u"我想听张学友的歌", u"周杰伦的龙卷风", u"鹿晗有什么歌好听", u"姚明打篮球好厉害", u"张继科会打乒乓球", u"詹姆士是体育明星"])
    #     Y_train = np.array([1, 1, 1, 2, 2, 2])
    #     Test_data = [u"我想听薛之谦的演员", "邓亚萍是体育明星", "刘翔是体育明星"]
    #     Model = Word2Vec()
    #     Model.load_W2V_Model("word2vec.model")
    #     Model.train(X_train, Y_train)
    #     Model.predict(Test_data)
    #
    #     Model.save_NBmodel("NB.model")
    #     del Model
    #
    #     NBmodel_test = Classify()
    #     NBmodel_test.load_NBmodel("NB.model")
    #     NBmodel_test.predict(Test_data)


    # word2Vec = Word2Vec()
    # matrix = word2Vec.get_w2v_model_matrix('蓝蓝的', w2v_model=con.model['w2v_size200'])

    # str1 = "蓝蓝的"
    # str2 = "蓝蓝的"
    # str3 = "蓝蓝的"
    # word2Vec = Word2Vec()
    # word2Vec.load_W2V_Model(con.model['w2v_size200'])
    # vec = word2Vec.sentence_2_vec([str1, str2, str3])
    # R1 = sum((vec[0] - vec[1]) ** 2) ** 0.5
    # R2 = sum((vec[0] - vec[2]) ** 2) ** 0.5
    # R3 = sum((vec[2] - vec[1]) ** 2) ** 0.5
    # print(R1)
    # print(R2)
    # print(R3)

    # pass
