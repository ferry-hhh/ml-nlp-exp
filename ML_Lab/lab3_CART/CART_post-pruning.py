#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File   :   CART_post-pruning.py
@Time   :   2021/11/18 12:07:08
@Author :   WYJ 
@Desc   :   构造一棵采用后剪枝技术的CART决策树，根据选取的tree_type不同可以进行分类任务和回归任务
            其中分类树和回归树中的函数基本都可以复用，但是为了代码的可读性与理解性分别进行了实现，内部逻辑类似
'''

import copy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


from node import Node

class CART:
    def __init__(self, data_set=None, discrete_attrs=None, tree_type=1, min_samples_leaf=2, max_deepth=9):
        """
        CART类的初始化函数，可以生成分类树与回归树（默认生成分类树）
        
        param data_set: 训练集的字典列表，以{属性:值}的形式存储，样本类别为最后一个键值对
        param discrete_attrs: 离散属性集合
        param tree_type: 构造树的类型 1为分类树，0为回归树
        param min_samples_leaf: 叶子节点的最小样本量，默认为2
        param max_deepth: 递归深度（树的深度），默认位9
        return: 得到一颗树的根节点
        """
        self.root = Node()
        self.data_set = {} if data_set is None else data_set
        self.attr_names = list(data_set[0].keys())[0:-1]
        self.discrete_attrs = [] if discrete_attrs is None else discrete_attrs
        self.tree_type = tree_type
        self.min_samples_leaf = min_samples_leaf
        self.max_deepth = max_deepth
        if tree_type == 1:
            self.grow_classify_CART()
        else:
            self.grow_regression_CART()


    def Gini(self, data_set_v):
        """
        计算输入数据集的基尼值
        
        param data_set_v: 输入数据集的字典列表，每一个样本对应一个dict，存储{属性:值}，所有样本组成一个list
        return: 基尼值
        """
        data_num = len(data_set_v)
        # 统计每个类别的数量
        type_cnt_dict = {} 
        for item in data_set_v:
            type_cnt_dict[item['target']] = \
                            type_cnt_dict.get(item['target'], 0) + 1
        # 基于公式4.5计算基尼值
        P_k_sum = 0
        for key in type_cnt_dict.keys():
            P_k_sum += pow(type_cnt_dict[key] / data_num, 2)
        gini_value = 1 - P_k_sum
        return gini_value


    def Gini_index(self, data_set, divide_attr):
        """
        计算输入属性的基尼指数
        
        param data_set: 当前节点数据集的字典列表 [{'sepal_length':7.7}, {'sepal_width':6.8}...{'target':1}]
        param divide_attr: 当前用于划分数据集的属性
        return: 输入属性的基尼指数与最优划分点
        """
        data_num = len(data_set)
        gini_index_value = 0 # 最终返回的基尼指数

        '''
        由于data_set是数据集的字典列表，首先要得到分割属性a对应的数据列表v_list
        遍历data_set，添加每一项a对应的值到v_list
        求出候选划分点集合
        对每个候选划分点t，求比t大的数据集和比t小的数据集
        对这两个数据集分别求基尼值，用公式4.6求解得到一个候选划分点的基尼指数
        对比所有候选划分点的基尼指数，选择最小的一个作为属性a的基尼指数
        类比离散的，连续的里面每个候选划分点分割开的2个数据集的地位和离散中的属性a对应的4个属性值求出的4个数据集地位相同
        而每个候选划分点相当于属性a的副本，最优的那个点的地位和属性a相同
        '''
        divide_attr_value_list = [] # 存储划分属性的值
        split_points = [] # 存储候选划分点
        split_res = [] # 存储候选划分结果
        # 统计划分属性的值
        for item in data_set:
            divide_attr_value_list.append(item[divide_attr])
        # 对划分属性的值进行排序
        sorted_attr_list = divide_attr_value_list.copy()
        sorted_attr_list.sort()
        # 统计候选划分点集合
        for i in range(data_num-1):
            split_points.append((sorted_attr_list[i]+sorted_attr_list[i+1])/2)
        
        s_split_points = set(split_points)
        # 求每个候选划分结果
        for point in s_split_points:
            data_set_v1 = list(filter(lambda item: item[divide_attr]<=point, data_set))
            data_set_v2 = list(filter(lambda item: item[divide_attr]>point, data_set))
            gini_index_value1 = (self.Gini(data_set_v1) * len(data_set_v1)) / data_num
            gini_index_value2 = (self.Gini(data_set_v2) * len(data_set_v2)) / data_num
            split_res.append(gini_index_value1 + gini_index_value2)
        # 求最优划分点与对应的基尼指数
        gini_index_value = min(split_res)
        split_value = split_points[split_res.index(gini_index_value)]
        return gini_index_value, split_value


    def get_best_divide_attr_classify(self, curr_node):
        """
        基于基尼指数从剩余属性中选择最优划分属性
        
        param curr_node: 当前节点
        return: 返回选择的划分属性与对应的划分值
        """
        attrs_gini_dict = {} # 统计每个属性的基尼指数
        attrs_split_value = {} # 统计每个属性的划分值
        remain_attrs = curr_node.remain_attrs # 从当前节点中得到剩余属性
        # 遍历每个剩余属性，求得基尼指数字典与划分值字典
        for attr in remain_attrs:
            data_set_a = self.split_data(curr_node.data_indexes)
            gini_value_a, split_value_a = self.Gini_index(data_set_a, attr)
            attrs_gini_dict[attr] = gini_value_a
            attrs_split_value[attr] = split_value_a
        # 选择基尼指数最小（优）的划分属性
        selected_attr = min(attrs_gini_dict, key=attrs_gini_dict.get)
        # 得到该划分属性对应的划分值
        split_value = attrs_split_value[selected_attr]
        return selected_attr, split_value


    def split_data(self, data_indexes):
        """
        辅助函数，按照样本索引列表从原数据集中摘出样本集
        
        param data_indexes: 某个节点拥有的样本索引列表
        return: 该样本索引列表对应的样本集
        """
        data_set = []
        for index in data_indexes:
            data_set.append(self.data_set[index])
        return data_set


    def build_classify_tree(self, curr_node, parent_leaf_type=0, deepth=1):
        """
        递归构造分类树算法
        
        param curr_node: 当前节点
        param parent_leaf_type: 父节点样本中最多的类别
        param deepth: 当前递归深度
        return: 最终构造一颗分类树
        """
        '''
        递归算法
        判断当前节点拥有的数据是否都是同一类别c，是则标记node为c类叶节点，然后就return
        判断待划分属性值是否空，或者是否当前数据属性值全都一样无法区分
            是则标记node为数据中最多的类别c，然后就return
        选择属性进行划分
        为选择的属性生成左孩子和右孩子
            如果某个孩子拥有的样本为空，那么就把父亲样本中最多的那个类别标记给该孩子node（因为实在没有数据，没办法继续判断了，只好将父样本中最多的类别当成孩子的类别）
            如果不为空，那么就调用build_tree递归
        '''
        # 得到当前节点的训练集
        data_indexes = curr_node.data_indexes
        # 判断当前节点样本集是否为空
        if len(data_indexes) == 0:
            curr_node.leaf_type = parent_leaf_type # 空则标记该孩子节点为父节点中最多的类别（叶节点）
            return
        node_data_set = self.split_data(data_indexes)
        # 得到当前训练集样本对应的类别
        node_data_set_targets = []
        for item in node_data_set:
            node_data_set_targets.append(item['target'])
        # 得到当前训练集样本最多的类别
        leaf_type = max(node_data_set_targets, key=node_data_set_targets.count)
        # 控制递归深度
        if deepth >= self.max_deepth:
            curr_node.leaf_type = leaf_type
            return
        # 判断当前节点拥有的样本是否都是同一类别
        if len(set(node_data_set_targets)) == 1: # 【！！！】忘记加len导致无法从同类节点return
            curr_node.leaf_type = node_data_set_targets[0] # 是则标记当前节点为该类别（叶节点）
            return

        # 判断待划分属性是否为空or样本属性值都一样（类别可能不一样）
        remain_attrs = copy.deepcopy(curr_node.remain_attrs) # 得到当前节点的剩余待划分属性
        # 判断待划分属性是否为空
        empty_flag = 0
        if len(remain_attrs) == 0:
            empty_flag = 1
        # 判断样本属性值是否都一样
        temp_node_data_set = copy.deepcopy(node_data_set) # 先复制一份数据
        for item in temp_node_data_set:
            del item['target'] # 因为类别可能不一样，就先删除类别的键值对
        temp_item = temp_node_data_set[0] # 便于比较
        same_flag = 1
        for item in temp_node_data_set:
            if item != temp_item: # 只要有一个不同，数据集中的样本就不一样
                same_flag = 0
        # 是则标记当前节点为最多样本的类别（叶节点）
        if empty_flag or same_flag:
            curr_node.leaf_type = leaf_type 
            return

        # 选择最优划分属性
        selected_attr, split_value = self.get_best_divide_attr_classify(curr_node)
        # 若被选择属性是离散属性，那么本次用完就移除
        if selected_attr in self.discrete_attrs:
            # 从剩余属性列表中移除被选择的属性
            remain_attrs.remove(selected_attr)
        # print(selected_attr)
        # 填充{划分属性:划分属性值}
        curr_node.attr_name2split_value[selected_attr] = split_value

        # 为选择的属性生成左孩子和右孩子
        # filter函数得到比划分值小、比划分值大的两部分数据集
        left_data_set_indexes = list(filter(lambda index: self.data_set[index][selected_attr] <= curr_node.attr_name2split_value[selected_attr], data_indexes))
        right_data_set_indexes = list(filter(lambda index: self.data_set[index][selected_attr] > curr_node.attr_name2split_value[selected_attr], data_indexes))
        # 左右孩子所划到的样本的属性值，左孩子的属性值为'<=划分值'
        left_attr_value = '<=' + str(curr_node.attr_name2split_value[selected_attr])
        right_attr_value = '>' + str(curr_node.attr_name2split_value[selected_attr])
        curr_node.left_child = Node(attr_value=left_attr_value, parent=curr_node, remain_attrs=remain_attrs, data_indexes=left_data_set_indexes)
        curr_node.right_child = Node(attr_value=right_attr_value, parent=curr_node, remain_attrs=remain_attrs, data_indexes=right_data_set_indexes)

        # 递归构建左右子树，同时深度+1，并且传参当前节点样本中最多的类别进入子节点的递归
        self.build_classify_tree(curr_node.left_child, leaf_type, deepth+1)
        self.build_classify_tree(curr_node.right_child, leaf_type, deepth+1)


    def grow_classify_CART(self):
        """
        生成一颗分类树
        
        return: 最终得到一颗分类树，可以沿着根节点走完整棵树
        """
        # 为数据集样本添加下标
        root_data_indexes = []
        for i in range(len(self.data_set)):
            root_data_indexes.append(i)
        # 初始化根节点
        root_node = Node(remain_attrs=copy.deepcopy(self.attr_names), data_indexes=root_data_indexes)
        self.root = root_node
        self.build_classify_tree(root_node, parent_leaf_type=0, deepth=1)


    def cal_mse(self, data_set_v):
        """
        计算输入数据集的mse
        
        param data_set_v: 输入数据集的字典列表，每一个样本对应一个dict，存储{属性:值}，所有样本组成一个list
        return: mse
        """
        data_num = len(data_set_v)
        # 统计每个类别的数量
        true_scores = []
        for item in data_set_v:
            true_scores.append(item['target'])
        mean_score = sum(true_scores) / data_num
        pred_scores = [mean_score] * data_num
        # 计算mse
        mse = mean_squared_error(true_scores, pred_scores) # 调用sklearn的MSE计算函数
        return mse


    def get_best_split_point(self, data_set, divide_attr):
        """
        计算输入属性的mse
        
        param data_set: 当前节点数据集的字典列表 [{'sepal_length':7.7}, {'sepal_width':6.8}...{'target':1}]
        param divide_attr: 当前用于划分数据集的属性
        return: 输入属性的mse与最优划分点
        """
        data_num = len(data_set)
        mse_value = 0 # 最终返回的mse

        divide_attr_value_list = [] # 存储划分属性的值
        split_points = [] # 存储候选划分点
        split_res = [] # 存储候选划分结果
        # 统计划分属性的值
        for item in data_set:
            divide_attr_value_list.append(item[divide_attr])
        # 对划分属性的值进行排序
        sorted_attr_list = divide_attr_value_list.copy()
        sorted_attr_list.sort()
        # 统计候选划分点集合
        for i in range(data_num-1):
            split_points.append((sorted_attr_list[i]+sorted_attr_list[i+1])/2)
        
        s_split_points = set(split_points)
        # 求每个候选划分结果
        for point in s_split_points:
            data_set_v1 = list(filter(lambda item: item[divide_attr]<=point, data_set))
            data_set_v2 = list(filter(lambda item: item[divide_attr]>point, data_set))
            mse_value1 = (self.Gini(data_set_v1) * len(data_set_v1)) / data_num
            mse_value2 = (self.Gini(data_set_v2) * len(data_set_v2)) / data_num
            split_res.append(mse_value1 + mse_value2)
        # 求最优划分点与对应的mse
        mse_value = min(split_res)
        split_value = split_points[split_res.index(mse_value)]
        return mse_value, split_value


    def get_best_divide_attr_reg(self, curr_node):
        """
        基于MSE从剩余属性中选择最优划分属性
        
        param curr_node: 当前节点
        return: 返回选择的划分属性与对应的划分值
        """
        attrs_mse_dict = {} # 统计每个属性的mse
        attrs_split_value = {} # 统计每个属性的划分值
        remain_attrs = curr_node.remain_attrs # 从当前节点中得到剩余属性
        # 遍历每个剩余属性，求得mse字典与划分值字典
        for attr in remain_attrs:
            data_set_a = self.split_data(curr_node.data_indexes)
            mse_value_a, split_value_a = self.get_best_split_point(data_set_a, attr)
            attrs_mse_dict[attr] = mse_value_a
            attrs_split_value[attr] = split_value_a
        # 选择mse最小（优）的划分属性
        selected_attr = min(attrs_mse_dict, key=attrs_mse_dict.get)
        # 得到该划分属性对应的划分值
        split_value = attrs_split_value[selected_attr]
        return selected_attr, split_value


    def build_regression_tree(self, curr_node, parent_mean_score, deepth):
        """
        递归构造回归树算法
        
        param curr_node: 当前节点
        param parent_mean_score: 父节点样本的均值
        param deepth: 当前递归深度
        return: 最终构造一颗回归树
        """
        '''
        每个叶节点的leaf_type现在存储预测值，也就是score
        这个分数是该节点样本或者父节点样本的socore均值
        递归时，先先选择最优划分属性，然后划分、接下来再递归子节点
        最优划分属性选择：
            先计算每个属性，每个划分点的均方差
            选最小均方差的属性
        '''
        # 得到当前节点的训练集
        data_indexes = curr_node.data_indexes

        # 判断当前节点样本集是否小于预定值
        if len(data_indexes) <= self.min_samples_leaf:
            curr_node.leaf_type = parent_mean_score
            return

        node_data_set = self.split_data(data_indexes)
        # 得到当前训练集样本对应的类别
        node_data_set_targets = []
        for item in node_data_set:
            node_data_set_targets.append(item['target'])

        # 得到当前节点样本集的样本平均分
        mean_score = sum(node_data_set_targets) / len(node_data_set_targets)
        if deepth >= self.max_deepth:
            curr_node.leaf_type = mean_score
            return

        # 判断待划分属性是否为空or样本属性值都一样（类别可能不一样）
        remain_attrs = copy.deepcopy(curr_node.remain_attrs) # 得到当前节点的剩余待划分属性
        # print(remain_attrs)
        # 判断待划分属性是否为空
        empty_flag = 0
        if len(remain_attrs) <= 1:
            empty_flag = 1
        # 判断样本属性值是否都一样
        temp_node_data_set = copy.deepcopy(node_data_set) # 先复制一份数据
        for item in temp_node_data_set:
            del item['target'] # 因为类别可能不一样，就先删除类别的键值对
        temp_item = temp_node_data_set[0] # 便于比较
        same_flag = 1
        for item in temp_node_data_set:
            if item != temp_item: # 只要有一个不同，数据集中的样本就不一样
                same_flag = 0
        # 是则标记当前节点为样本平均分（叶节点）
        if empty_flag or same_flag:
            curr_node.leaf_type = mean_score
            return

        # 选择最优划分属性
        selected_attr, split_value = self.get_best_divide_attr_reg(curr_node)
        # 从剩余属性列表中移除被选择的属性
        remain_attrs.remove(selected_attr)
        # print(selected_attr)
        # 填充{划分属性:划分属性值}
        curr_node.attr_name2split_value[selected_attr] = split_value

        # 为选择的属性生成左孩子和右孩子
        # filter函数得到比划分值小、比划分值大的两部分数据集
        left_data_set_indexes = list(filter(lambda index: self.data_set[index][selected_attr] <= curr_node.attr_name2split_value[selected_attr], data_indexes))
        right_data_set_indexes = list(filter(lambda index: self.data_set[index][selected_attr] > curr_node.attr_name2split_value[selected_attr], data_indexes))
        # 左右孩子所划到的样本的属性值，左孩子的属性值为'<=划分值'
        left_attr_value = '<=' + str(curr_node.attr_name2split_value[selected_attr])
        right_attr_value = '>' + str(curr_node.attr_name2split_value[selected_attr])
        curr_node.left_child = Node(attr_value=left_attr_value, parent=curr_node, remain_attrs=remain_attrs, data_indexes=left_data_set_indexes)
        curr_node.right_child = Node(attr_value=right_attr_value, parent=curr_node, remain_attrs=remain_attrs, data_indexes=right_data_set_indexes)

        # 递归构建左右子树，同时深度+1，并且传参当前节点样本的均值进入子节点的递归
        self.build_regression_tree(curr_node.left_child, mean_score, deepth+1)
        self.build_regression_tree(curr_node.right_child, mean_score, deepth+1)


    def grow_regression_CART(self):
        """
        生成一颗回归树
        
        return: 最终得到一颗回归树，可以沿着根节点走完整棵树
        """
        root_data_indexes = []
        for i in range(len(self.data_set)):
            root_data_indexes.append(i)
        root_node = Node(remain_attrs=copy.deepcopy(self.attr_names), data_indexes=root_data_indexes)
        self.root = root_node
        self.build_regression_tree(root_node, parent_mean_score=0, deepth=1)


    def post_pruning(self, test_dict_list):
        """
        生成一棵树之后进行后剪枝，且以尽量精简树的结构为目标
        
        param test_dict_list: 测试集
        return: 得到一个剪枝后的树的根节点
        """
        print('剪枝中...')
        nodes_waiting_judge = [] # 待剪枝节点列表
        node_queue = [] # 节点队列
        node_queue.append(self.root)
        # 首先使用广度优先遍历将最后一层分支节点加入待剪枝列表
        while len(node_queue) > 0:
            curr_node = node_queue[0]
            # 首先判断是分支节点
            if curr_node.leaf_type is None:
                node_queue.append(curr_node.left_child)
                node_queue.append(curr_node.right_child)
                # 其次判断左右孩子都是叶节点
                if curr_node.left_child.leaf_type and curr_node.right_child.leaf_type:
                    nodes_waiting_judge.append(curr_node)
            node_queue.remove(curr_node)
        # 进行剪枝
        while len(nodes_waiting_judge) > 0:
            curr_node = nodes_waiting_judge.pop() # 获取当前节点，同时从列表中pop出去
            # 得到剪枝前的准确率orR2值
            if self.tree_type:
                acc_before_prune = self.test_classify(test_dict_list)
            else:
                acc_before_prune = self.test_reg_R2(test_dict_list)
            node_data_set = self.split_data(curr_node.data_indexes)
            # 得到当前训练集样本对应的类别
            node_data_set_targets = []
            for item in node_data_set:
                node_data_set_targets.append(item['target'])
            # 得到剪枝后的准确率orR2值
            if self.tree_type:
                leaf_type = max(node_data_set_targets, key=node_data_set_targets.count)
                curr_node.leaf_type = leaf_type # 这里注意，在test函数中判断是否到达叶子节点使用leaf_type而不是attr_name2split_value
                acc_after_prune = self.test_classify(test_dict_list)
            else:
                mean_score = sum(node_data_set_targets) / len(node_data_set_targets)
                curr_node.leaf_type = mean_score
                acc_after_prune = self.test_reg_R2(test_dict_list)
            # 判断是否剪枝
            if acc_before_prune > acc_after_prune: # 不剪枝
                curr_node.leaf_type = None # 回退leaf_type值
            else: # 剪枝
                curr_node.attr_name2split_value = {} # 保持叶子节点的属性同步
                curr_node.left_child = None
                curr_node.right_child = None
                # 判断父节点是否也要加入待剪枝列表
                parent_node = curr_node.parent
                if parent_node.left_child.leaf_type and parent_node.right_child.leaf_type:
                    nodes_waiting_judge.append(parent_node)


    def print_tree(self):
        """
        打印这棵树以便直观的观察分类规则
        """
        # 使用队列，以广度优先的方法打印每一层的节点
        node_queue = [copy.deepcopy(self.root)]
        node_cnt = 1
        while(len(node_queue)>0):
            node_cnt += 1
            curr_node = node_queue[0]
            # print(curr_node.node_to_string()) # 调用Node的描述方法
            # 不能用if not curr_node.leaf_type: 因为0也是其中一个类型，这样会把叶子节点误判为分支节点
            if curr_node.leaf_type is None:
                node_queue.append(curr_node.left_child)
                node_queue.append(curr_node.right_child)
            node_queue.remove(curr_node)
            # print('---------------------------------------')
        print('节点总数为：', node_cnt)


    def test_classify(self, test_dict_list):
        """
        对输入的测试集调用决策树进行预测分类
        
        param test_dict_list: 测试集，和训练集一样是字典列表
        return acc: 准确率
        """
        target = []
        pre_target = []
        cnt = 0
        for item in test_dict_list:
            curr_node = copy.deepcopy(self.root)
            while curr_node.leaf_type is None:
                if item[list(curr_node.attr_name2split_value.keys())[0]] <= list(curr_node.attr_name2split_value.values())[0]:
                    curr_node = curr_node.left_child
                else:
                    curr_node = curr_node.right_child
            if item['target'] == curr_node.leaf_type:
                cnt += 1
            target.append(item['target'])
            pre_target.append(curr_node.leaf_type)
        acc = cnt / len(test_dict_list)
        # print('实际类别：', target)
        # print('预测类别：', pre_target)
        # print('准确率=', acc)
        return acc


    def test_reg_R2(self, test_dict_list):
        """
        对输入的测试集调用决策树进行预测回归
        
        param test_dict_list: 测试集，和训练集一样是字典列表
        return R2: R2值，越大越好，最大为1
        """
        target = []
        pre_target = []
        for item in test_dict_list:
            curr_node = copy.deepcopy(self.root)
            while curr_node.leaf_type is None:
                if item[list(curr_node.attr_name2split_value.keys())[0]] <= list(curr_node.attr_name2split_value.values())[0]:
                    curr_node = curr_node.left_child
                else:
                    curr_node = curr_node.right_child
            target.append(item['target'])
            pre_target.append(curr_node.leaf_type)
        mse = mean_squared_error(target, pre_target)
        R2=1-mse/np.var(target)
        # print('R2=', R2)
        return R2



def DFS(node, s='root'):
    """
    工具类，用于深度优先打印一棵树，检查树的生成是否有问题
    例如叶子节点的划分属性和类别同时为空，就有问题
    
    param node: 节点
    param s: 用于打印路径，默认在root
    """
    if node == None:
        return
    print('划分：', node.attr_name2split_value, '\t属性：', node.leaf_type, '\t属性值：', node.attr_value)
    if node.attr_name2split_value == {} and node.leaf_type != None:
        s += '：叶子'
        print('一条路径完毕')
    if node.attr_name2split_value == {} and node.leaf_type == None:
        print('【叶节点生成错误】叶节点的划分属性与叶子类型不可同时为空！')
    print(s, '\n----------------------------------------')
    DFS(node.left_child, s+'->left')
    DFS(node.right_child, s+'->right')


def load(data, attributes):
    """
    构造字典数据集
    
    param data: 二维列表，存储每一行数据
    param attributes: 属性种类列表
    return: 一个字典列表，每一行数据以字典的方式将属性和值对应起来
    """
    dict_list = []
    for item in data:
        item_dict = {}
        for i in range(len(attributes)):
            item_dict[attributes[i]] = item[i]
        dict_list.append(item_dict)
    return dict_list


def classify_iris_test():
    """
    测试决策树在鸢尾花数据集的准确度
    """
    def load_iris_data():
        """
        加载鸢尾花数据集
        
        return: 训练字典列表与对应的类别，测试字典列表与对应的类别
        """
        iris = load_iris()
        # 以3:1划分训练集与测试集
        train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.2)
        train_data = train_data.tolist()
        test_data = test_data.tolist()
        for i in range(len(train_data)):
            train_data[i].append(train_target[i])
        for i in range(len(test_data)):
            test_data[i].append(test_target[i])

        attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
        train_dict_list = load(train_data, attributes)
        test_dict_list = load(test_data, attributes)
        return train_dict_list, test_dict_list

    print('正在测试鸢尾花三分类任务...')
    # 得到训练集和测试集
    train_dict_list, test_dict_list = load_iris_data()
    # 训练分类树
    cart = CART(train_dict_list)
    # 层层打印这棵树
    cart.print_tree()
    # 输入测试集进行测试
    acc_before_prune = cart.test_classify(test_dict_list)
    print('准确率=', acc_before_prune)
    cart.post_pruning(test_dict_list)
    print('******************************剪枝后******************************')
    cart.print_tree()
    acc_after_prune = cart.test_classify(test_dict_list)
    print('准确率=', acc_after_prune)
    # DFS(cart.root, 'root')


def classify_melon_test():
    """
    西瓜数据集，为了测试离散属性的处理
    """
    def load_melon_data():
        '''
        离散属性数字化
        色泽：{浅白：1}，{青绿：2}，{乌黑：3}
        根蒂：{硬挺：1}，{稍蜷：2}，{蜷缩：3}
        敲声：{清脆：1}，{浊响：2}，{沉闷：3}
        纹理：{清晰：1}，{稍糊：2}，{模糊：3}
        脐部：{平坦：1}，{稍凹：2}，{凹陷：3}
        触感：{硬滑：1}，{软粘：2}
        '''
        train_data = [
            [2, 3, 2, 1, 3, 1, '是'],
            [3, 3, 3, 1, 3, 1, '是'],
            [3, 3, 2, 1, 3, 1, '是'],
            [2, 2, 2, 1, 2, 2, '是'],
            [3, 2, 2, 2, 2, 2, '是'],
            [2, 1, 1, 1, 1, 2, '否'],
            [1, 2, 3, 2, 3, 1, '否'],
            [3, 2, 2, 1, 2, 2, '否'],
            [1, 3, 2, 3, 1, 1, '否'],
            [2, 3, 3, 2, 2, 1, '否']
        ]
    
        test_data = [
            [2, 3, 3, 1, 3, 1, '是'],
            [1, 3, 2, 1, 3, 1, '是'],
            [3, 2, 2, 1, 2, 1, '是'],
            [3, 2, 3, 2, 2, 1, '否'],
            [1, 1, 1, 3, 1, 1, '否'],
            [1, 3, 2, 3, 1, 2, '否'],
            [2, 2, 2, 2, 3, 1, '否'],
        ]


        attributes = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', 'target']
        train_dict_list = load(train_data, attributes)
        test_dict_list = load(test_data, attributes)
        discrete_attrs = attributes[0:-1]
        return train_dict_list, test_dict_list, discrete_attrs
    print('正在测试西瓜二分类任务...')
    # 得到训练集和测试集
    train_dict_list, test_dict_list, discrete_attrs = load_melon_data()
    # 训练分类树
    cart = CART(train_dict_list, discrete_attrs)
    # 层层打印这棵树
    cart.print_tree()
    # 输入测试集进行测试
    acc_before_prune = cart.test_classify(test_dict_list)
    print('准确率=', acc_before_prune)
    cart.post_pruning(test_dict_list)
    print('******************************剪枝后******************************')
    cart.print_tree()
    acc_after_prune = cart.test_classify(test_dict_list)
    print('准确率=', acc_after_prune)


def classify_wine_test():
    """
    测试决策树在红酒数据集的准确度
    """
    def load_wine_data():
        """
        加载红酒数据集
        
        return: 训练字典列表与对应的类别，测试字典列表与对应的类别
        """
        wine = load_wine()
        # 以3:1划分训练集与测试集
        train_data, test_data, train_target, test_target = train_test_split(wine.data, wine.target, test_size=0.2)
        train_data = train_data.tolist()
        test_data = test_data.tolist()
        for i in range(len(train_data)):
            train_data[i].append(train_target[i])
        for i in range(len(test_data)):
            test_data[i].append(test_target[i])

        attributes = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','稀释葡萄酒','脯氨酸', 'target']
        train_dict_list = load(train_data, attributes)
        test_dict_list = load(test_data, attributes)
        return train_dict_list, test_dict_list

    print('正在测试红酒三分类任务...')
    # 得到训练集和测试集
    train_dict_list, test_dict_list = load_wine_data()
    # 训练分类树
    cart = CART(train_dict_list)
    # 层层打印这棵树
    cart.print_tree()
    # 输入测试集进行测试
    acc_before_prune = cart.test_classify(test_dict_list)
    print('准确率=', acc_before_prune)
    cart.post_pruning(test_dict_list)
    print('******************************剪枝后******************************')
    cart.print_tree()
    acc_after_prune = cart.test_classify(test_dict_list)
    print('准确率=', acc_after_prune)


def regression_boston_test():
    """
    测试决策树在波士顿房价数据集的准确度
    """
    def load_boston_data():
        """
        加载波士顿房价数据集
        
        return: 训练字典列表与对应的类别，测试字典列表与对应的类别
        """
        boston = load_boston()
        # 以3:1划分训练集与测试集
        train_data, test_data, train_target, test_target = train_test_split(boston.data, boston.target, test_size=0.2)
        train_data = train_data.tolist()
        test_data = test_data.tolist()
        for i in range(len(train_data)):
            train_data[i].append(train_target[i])
        for i in range(len(test_data)):
            test_data[i].append(test_target[i])

        attributes = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'target',]
        train_dict_list = load(train_data, attributes)
        test_dict_list = load(test_data, attributes)
        return train_dict_list, test_dict_list

    print('正在测试波士顿房价回归任务...')
    # 得到训练集和测试集
    train_dict_list, test_dict_list = load_boston_data()
    # 训练分类树
    cart = CART(train_dict_list, tree_type=0)
    # 层层打印这棵树
    cart.print_tree()
    cart.test_reg_R2(test_dict_list)
    # 输入测试集进行测试
    acc_before_prune = cart.test_reg_R2(test_dict_list)
    print('R2=', acc_before_prune)
    cart.post_pruning(test_dict_list)
    print('******************************剪枝后******************************')
    cart.print_tree()
    acc_after_prune = cart.test_reg_R2(test_dict_list)
    print('R2=', acc_after_prune)
    # DFS(cart.root, 'root')


def regression_diabetes_test():
    """
    测试决策树在糖尿病数据集的准确度
    """
    def load_diabetes_data():
        """
        加载糖尿病数据集
        
        return: 训练字典列表与对应的类别，测试字典列表与对应的类别
        """
        diabetes = load_diabetes()
        # 以3:1划分训练集与测试集
        train_data, test_data, train_target, test_target = train_test_split(diabetes.data, diabetes.target, test_size=0.2)
        train_data = train_data.tolist()
        test_data = test_data.tolist()
        for i in range(len(train_data)):
            train_data[i].append(train_target[i])
        for i in range(len(test_data)):
            test_data[i].append(test_target[i])

        attributes = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'target',]
        train_dict_list = load(train_data, attributes)
        test_dict_list = load(test_data, attributes)
        return train_dict_list, test_dict_list

    print('正在测试糖尿病回归任务...')
    # 得到训练集和测试集
    train_dict_list, test_dict_list = load_diabetes_data()
    # 训练分类树
    cart = CART(train_dict_list, tree_type=0)
    # 层层打印这棵树
    cart.print_tree()
    cart.test_reg_R2(test_dict_list)
    # 输入测试集进行测试
    acc_before_prune = cart.test_reg_R2(test_dict_list)
    print('R2=', acc_before_prune)
    cart.post_pruning(test_dict_list)
    print('******************************剪枝后******************************')
    cart.print_tree()
    acc_after_prune = cart.test_reg_R2(test_dict_list)
    print('R2=', acc_after_prune)


if __name__ == "__main__":
    # classify_iris_test()
    # classify_melon_test()
    # classify_wine_test()
    regression_boston_test()
    # regression_diabetes_test()