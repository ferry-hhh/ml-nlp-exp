#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   test_HMM.py
@Time   :   2021/11/12 23:22:59
@Author :   WYJ 
@Desc   :   输入测试集测试预测准确率
'''

import os
from pinyin2hanzi import HMM

root = os.path.abspath('.')
test_path = str(root) + r"\lab1_data\测试集.txt"
# 读取测试集
with open(test_path, 'r') as f:
    lins = f.readlines()

# 初始化HMM模型
hmm = HMM()

# 分离拼音列表与语句列表
pinyin_list = []
sen_true_list = []
for i in range(len(lins)):
    if i % 2 == 0:
        pinyin = lins[i].lower().strip('\n') # 将拼音全部转换小写
        pinyin_list.append(pinyin)
    else:
        sen_true_list.append(lins[i].strip('\n'))

sen_num = len(sen_true_list)
test_word_num = 0
right_word_num = 0
right_sen = 0
f = open('result_new.txt', 'w', encoding='utf-8')
for i in range(sen_num):
    sen_pred = hmm.viterbi(pinyin_list[i]) # 进行预测
    # 将结果写入result文件
    f.write('拼音输入：' + pinyin_list[i] + '\n')
    f.write('真实语句：' + sen_true_list[i] + '\n')
    f.write('预测语句：' + sen_pred + '\n')

    right_num = 0
    # 采用严格标准检查句子是否完全正确，分别统计字正确率与句子正确率
    flag = 1
    for j in range(len(sen_true_list[i])):
        test_word_num += 1
        if sen_true_list[i][j] != sen_pred[j]:
            flag = 0
        if sen_true_list[i][j] == sen_pred[j]:
            right_num += 1
            right_word_num += 1
    if flag == 1:
        right_sen += 1

    f.write('该句的单字正确率为：' + str(right_num / len(sen_true_list[i])) + '\n')
f.write('*********************************************************\n')
f.write('完全正确的句子个数：' + str(right_sen) + '\n')
f.write('字的正确率为：' + str(right_word_num / test_word_num) + '\n')
f.write('句子的正确率：' + str(right_sen / sen_num) + '\n')
print('测试完毕！完整测试结果见 result_new.txt')
print('完全正确的句子个数：', right_sen)
print('字的正确率为：', right_word_num / test_word_num)
print('句子的正确率：', right_sen / sen_num)