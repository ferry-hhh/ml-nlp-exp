#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   preprocess_data.py
@Time   :   2021/11/10 13:38:25
@Author :   WYJ 
@Desc   :   对训练数据集进行预处理，包括清洗数据集、生成初始概率矩阵JSON、
            生成转移概率矩阵JSON、生成发射概率矩阵JSON
'''


import os
import re
import math
import json
import pypinyin
root = os.path.abspath('.')
raw_data_path = str(root) + r"\lab1_data\toutiao_cat_data.txt"
filter_data_path = "./filter.txt"
def my_filter(file_path):
    """
    过滤原始数据集中的英文字母、数字、标点等

    param file_path: 原始数据文件路径
    return: 输出一个filter.txt的干净文件
    """
    with open('special_symbols.txt', 'r', encoding='utf-8') as f:
        ss = f.read()
    f1 = open("filter.txt", "w", encoding="utf-8")
    for line in open(file_path, encoding="utf-8"):
        line = line.replace(',', '，')
        punctuations = r"[a-zA-Z0-9_!_" + ss +"]+"
        new_line = re.sub(punctuations, "", line) # 将字母、数字、特殊字符替换为空
        f1.write(new_line)
    f1.close()
    print(file_path + " 过滤完毕！请查看filter.txt文件")


def load_data_list():
    """
    将filter.txt文件中的语句按照标点分割成句，存储在列表中
    
    return: 返回一个列表，存储每一句话
    """
    with open("./filter.txt", 'r', encoding='utf-8') as f:
        lines = f.read()
        lines = re.sub(r"\n", "", lines)
        punctuations = r'⋯|\'|？|：| |“|”|\.|/|《|》|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！|…|┈|（|）|\t'
        sens_list = re.split(punctuations, lines) # 按照标点分割。注意此处使用re的split方法，故匹配字符串要加更多的转义字符
    # print(sens_list[:10])
    return sens_list


def save(file_name, data):
    """
    将数据写入json文件
    
    param file_name: 文件名
    param data: 数据
    """
    with open(file_name + ".json", "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2) # 保存中文，缩进两格
    
def build_init(sens_list):
    """
    统计并构造汉字初始概率矩阵（即计算每个汉字出现在句首的概率）
    {字 : 概率}
    
    param sens_list: 存储每一句话的列表
    return: 构造一个init_prob.json文件
    """
    print('Building initial probability...')
    init_prob = {}
    num = 0
    total = len(sens_list)
    for sen in sens_list:
        if not sen.isspace(): # 防止读到空句子
            init_prob[sen[0]] = init_prob.get(sen[0], 0) + 1 # 统计句首汉字个数
            # 记录统计进程
            num += 1
            if not num % 100000:
                print("{}/{}".format(num, total))
    
    for key in init_prob.keys():
        init_prob[key] = math.log(init_prob.get(key) / total) # 计算并用log函数规范化概率
    
    save('init_porb', init_prob) # 保存为json文件
    print('********************build_init succeed!********************')


def build_launch(sens_list):
    """
    统计并构造汉字(隐序列)->拼音(观测序列)的发射概率矩阵
    即调用pypinyin对每句话进行拼音标注，记录每个汉字可能对应的拼音种类和个数【多音字而言】
    计算一个多音字发射到某个拼音的概率
    {字 : {拼音 : 概率}}

    param sens_list: 存储每一句话的列表
    return: 构造一个launch_prob.json文件
    """
    print('Building launch probability...')
    launch_prob = {}
    num = 0
    total = len(sens_list)
    for sen in sens_list:
        if not sen.isspace():
            pinyin = pypinyin.lazy_pinyin(sen) # 为每句话标注拼音
            for py, word in zip(pinyin, sen): # 同时迭代拼音和汉字
                if not launch_prob.get(word, None): # 如果没有这个汉字
                    launch_prob[word] = {} # 那么就新建一个该汉字对应的拼音字典
                launch_prob[word][py] = launch_prob[word].get(py, 0) + 1 # 统计该汉字对应的拼音种类和个数
        num += 1
        if not num % 100000:
            print("{}/{}".format(num, total))
    # print(launch_prob)
    for word in launch_prob.keys():
        total_word2pys = sum(launch_prob.get(word).values()) # 获得单个汉字对应拼音数量之和
        for key in launch_prob.get(word):
            launch_prob[word][key] = math.log(launch_prob[word][key] / total_word2pys)
    
    save('launch_prob', launch_prob)
    print('********************build_launch succeed!********************')


def build_trans(sens_list):
    """
    统计并构造汉字之间的转移概率矩阵
    类似bigram模型
    {字 : {前一个字 : 概率}}
    
    param sens_list: 存储每一句话的列表:
    return: 构造一个trans_prob.json文件
    """
    print('Building transition probability...')
    trans_prob = {}
    num = 0
    total = len(sens_list)
    for sen in sens_list:
        if not sen.isspace():
            sen = [w for w in sen]
            sen.insert(0, 'BOS')
            sen.append('EOS')
            for index, word in enumerate(sen):
                if index:
                    pre_word = sen[index - 1]
                    if not trans_prob.get(word, None):
                        trans_prob[word] = {}
                    trans_prob[word][pre_word] = trans_prob[word].get(pre_word, 0) + 1
        num += 1
        if not num % 100000:
            print('{}/{}'.format(num, total))
    
    for word in trans_prob.keys():
        total_wordD = sum(trans_prob.get(word).values())
        for pre_word in trans_prob.get(word).keys():
            trans_prob[word][pre_word] = math.log(trans_prob[word].get(pre_word) / total_wordD)
    
    save('trans_prob', trans_prob)
    print('********************build_trans succeed!********************')


def build_same_pinyin_word():
    """
    统计同音字
    {拼音 : [字1, 字2]}
    
    return: 构造一个pinyin2words.json文件
    """
    print('Building pinyin status...')
    with open('launch_prob.json', 'r', encoding='utf-8') as f:
        launch_prob = json.load(f)
    
    data = {}
    for key in launch_prob.keys():
        for pinyin in launch_prob.get(key):
            if not data.get(pinyin, None):
                data[pinyin] = []
            data[pinyin].append(key)
    save('pinyin2words', data)
    print('********************build_same_pinyin_word succeed!********************')
    
my_filter(raw_data_path)
sens_list = load_data_list()
build_init(sens_list)
build_launch(sens_list)
build_trans(sens_list)
build_same_pinyin_word()
