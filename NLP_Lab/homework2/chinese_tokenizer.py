#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   chinese_tokenizer.py
@Time   :   2021/11/22 09:15:55
@Author :   WYJ 
@Desc   :   作业二：使用前向、后向算法实现一个简单中文分词器，开发语言不限。
'''

import re

class ChineseTokenizer:
    def __init__(self, file_path):
        """
        中文分词器，包括FMM与BMM算法
        
        param file_path: 原中文词典文件路径
        """
        self.file_path = file_path
        self.words_set, self.max_words_len = self.load_data_set() # 加载中文词典


    def load_data_set(self):
        """
        使用set加载中文词典
        
        param file_path: 词典文件路径
        return words_set: 存储在set中的词典库
        return max_words_len: 词典中最长的词语长度
        """
        print('词典加载中...')
        words_set = set() # 使用set来构建词典
        max_words_len = 0 # 维护一个词典中最大词长度，方便算法分词
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for words in f.readlines():
                words = words.strip('\n')
                new_words = re.sub(r'[0-9a-zA-Z]+', '', words)
                words_set.add(new_words)
                max_words_len = max(max_words_len, len(new_words))
        return words_set, max_words_len


    def FMM(self, sentence):
        """
        正向最大匹配算法，正向移动，缩短子串时删除末尾的字
        
        param sentence: 待分词句子
        return res: 分词列表
        """
        res = []
        # 外层循环切句子
        while len(sentence) > 0:
            # 初始化一个切词划分窗口，取词典最长词长度和句长的最小值
            split_win_len = min(self.max_words_len, len(sentence))
            # 初始化词
            sub_sen = sentence[0:split_win_len] # 正向移动
            # 内层循环匹配词
            while len(sub_sen) > 0:
                # 如果词在词典中
                if sub_sen in self.words_set:
                    res.append(sub_sen)
                    break
                # 如果词长度为1，说明词典中没有当前词
                elif len(sub_sen) == 1:
                    res.append(sub_sen)
                    break
                # 如果当前词在词典中没有，那么删除末尾的字，再进入while匹配
                else:
                    split_win_len -= 1 # 划分窗口缩短
                    sub_sen = sub_sen[0:split_win_len] # 子串尾部pop
            # 一次匹配结束更新原句子
            sentence = sentence[split_win_len:]
        return res


    def BMM(self, sentence):
        """
        逆向最大匹配算法，逆向移动，缩短子串时删除首部的字
        
        param :
        return:
        """
        res = []
        while len(sentence) > 0:
            split_win_len = min(self.max_words_len, len(sentence))
            sub_sen = sentence[-split_win_len:]
            while len(sub_sen) > 0:
                if sub_sen in self.words_set: # 逆向移动
                    res.append(sub_sen)
                    break
                elif len(sub_sen) == 1:
                    res.append(sub_sen)
                    break
                else:
                    split_win_len -= 1 # 划分窗口缩短
                    sub_sen = sub_sen[-split_win_len:] # 子串首部pop
            sentence = sentence[0:-split_win_len]
        res = res[::-1]
        return res


tokenizer = ChineseTokenizer(r'homework2\111186763ciku\中文分词词库整理\dict.txt')
f1 = open(r'homework2\test_result.txt', 'w', encoding='utf-8')
with open(r'homework2\test.txt', 'r', encoding='utf-8') as f:
    for sentence in f.readlines():
        f1.write('待分词语句为：' + sentence)
        sentence = sentence.strip('\n')
        res_FMM = tokenizer.FMM(sentence)
        res_BMM = tokenizer.BMM(sentence)
        f1.write('FMM分词结果：' + str(res_FMM) + '\n')
        f1.write('BMM分词结果：' + str(res_BMM) + '\n')
f1.close()
