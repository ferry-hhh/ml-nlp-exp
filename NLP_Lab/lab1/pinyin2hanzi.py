#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   PinYin2HanZi.py
@Time   :   2021/11/10 13:41:45
@Author :   WYJ 
@Desc   :   读取三种概率矩阵并使用viterbi算法进行拼音转汉字
'''

import json

class HMM:
    def __init__(self):
        self.load_param()
        self.min_p = -3.14e+100


    def read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


    def load_param(self):
        """
        加载json文件，生成初始概率矩阵、发射概率矩阵、转移概率矩阵
        """
        self.init_prob = self.read_json('init_porb.json')           # {字 : 概率}
        self.launch_prob = self.read_json('launch_prob.json')       # {字 : {拼音 : 概率}}
        self.trans_prob = self.read_json('trans_prob.json')         # {字 : {前一个字 : 概率}}
        self.pinyin2words = self.read_json('pinyin2words.json')     # {拼音 : [字1, 字2]}


    def viterbi(self, pinyin):
        """
        根据输入的拼音使用viterbi算法求解最优汉字序列
        因为计算三个矩阵的时候，已经使用log规范化数据，所以现在的概率相乘全部变成概率相加
        每一个拼音的所有同音字组成了viterbi算法中篱笆网络每一层的节点

        param pinyin: 输入的拼音
        return: 一句话
        """
        pinyin = pinyin.split()
        length = len(pinyin)
        viterbi = {}                # {0 : {字 : (概率, 前一个字)}}
        for i in range(length):
            viterbi[i] = {} # 内部的字典，键为所有同音字，值为(概率，前一个字)的元组

        # 处理拼音序列第一个字 对应篱笆网络的第一层
        for w in self.pinyin2words.get(pinyin[0]):
            w_init = self.init_prob.get(w, self.min_p)
            w_launch = self.launch_prob.get(w, {}).get(pinyin[0], self.min_p)
            w_trans = self.trans_prob.get(w, {}).get('BOS', self.min_p)
            viterbi[0][w] = (w_init + w_launch + w_trans, -1)

        # 处理拼音序列中间的字，对应篱笆网络中间层（不处理最后一层）
        for i in range(length - 1):
            for w in self.pinyin2words.get(pinyin[i+1]):
                viterbi_temp = []
                for pre_w in self.pinyin2words.get(pinyin[i]):
                    w_pre_prob = viterbi[i][pre_w][0]
                    w_launch = self.launch_prob.get(w, {}).get(pinyin[i+1], self.min_p)
                    w_trans = self.trans_prob.get(w, {}).get(pre_w, self.min_p)
                    viterbi_temp.append((w_pre_prob + w_launch + w_trans, pre_w))
                viterbi[i+1][w] = max(viterbi_temp)

        # 处理最后一个拼音，对应篱笆网络最后的一层
        for w in self.pinyin2words.get(pinyin[-1]):
            w_pre_prob = viterbi[length-1][w][0]
            w_launch = self.launch_prob.get(w, {}).get(pinyin[-1], self.min_p)
            w_trans = self.trans_prob.get('EOS', {}).get(w, self.min_p)
            viterbi[length-1][w] = (w_pre_prob + w_trans + w_launch, viterbi[length-1][w][1])

        words = [None] * length
        words[-1] = max(viterbi[length-1], key=viterbi[length-1].get)
        for n in range(length-2, -1, -1):
            words[n] = viterbi[n+1][words[n+1]][1]
        
        return ''.join(w for w in words)

if __name__ == "__main__":
    hmm = HMM()
    print(hmm.viterbi('chong qing da xue ji suan ji ke xue yu ji shu'))
    print(hmm.viterbi('wo ai wo jia'))