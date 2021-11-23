#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   node.py
@Time   :   2021/11/18 10:49:33
@Author :   WYJ 
@Desc   :   构建一个决策树的节点类
'''


class Node:
    def __init__(self, attr_name2split_value=None, attr_value=None, parent=None, left_child=None, right_child=None,
                 leaf_type=None, remain_attrs=None, data_indexes=None):
        """
        节点类初始化函数
        节点类别标识：
                    分支节点：attr_name2split_value != {} and leaf_type == None
                    叶子节点：attr_name2split_value == {} and leaf_type != None
                    判断叶子节点优先使用leaf_type，防止忘记更新attr_name2split_value导致出错
        
        param attr_value: 本节点中样本的属性值，与父节点的属性名相对应。
        param attr_name2split_value: 本节点进行划分子节点的属性名与值的字典 {name:value}
        param parent: 本节点的父节点
        param left_child: 本节点的左孩子
        param right_child: 本节点的右孩子
        param leaf_type: 本节点若是叶节点，该值标识样本类型
        param remain_attrs: 剩余未划分的属性
        param data_indexes: 划分到本节点的样本数据下标列表
        """
        self.attr_value = attr_value
        self.attr_name2split_value = {} if attr_name2split_value is None else attr_name2split_value
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        self.leaf_type = leaf_type
        self.remain_attrs = [] if remain_attrs is None else remain_attrs
        self.data_indexes = [] if data_indexes is None else data_indexes


    def node_to_string(self):
        """
        将节点相关属性用字符串描述
        
        return: 返回描述字符串
        """
        info = ''
        if self.leaf_type is None:
            if self.parent == None:
                info += '【根节点】'
                info += '【分支节点】：\n划分属性：' + str(list(self.attr_name2split_value.keys())[0])\
                        + '\t划分点：' + str(list(self.attr_name2split_value.values())[0]) + '\n'
            else:
                info += '【分支节点】：\n'
                info += '所属分支：' + str(list(self.parent.attr_name2split_value.keys())[0])\
                        + '(' + self.attr_value + ')' + '\n'
                info += '划分属性：' + str(list(self.attr_name2split_value.keys())[0])\
                        + '\t划分点：' + str(list(self.attr_name2split_value.values())[0]) + '\n'
        else:
            info += '【叶节点】：' + '\n'
            info += '所属分支：' + str(list(self.parent.attr_name2split_value.keys())[0])\
                    + '(' + self.attr_value + ')' + '\n'
            info += '类别：' + str(self.leaf_type) + '\n'
        return info
        