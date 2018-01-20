# -*- coding: utf8 -*-
# decision tree implementation

from math import log
import numpy as np


def create_dataset():
    """create some data for test"""
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels

def calc_shanon_ent(dataSet):
    """formula for entrophy: Info(D) = -\\sum_{i=1}^{m} p_i*log(p_i)"""
    numOfSample = len(dataSet)
    # cal how many feature the sample have
    labelCount = {}
    for feats in dataSet:
        label = feats[-1]
        # record each label frequency
        if label not in labelCount:
            labelCount[label] = 0
            labelCount[label] += 1
    shanon_ent = 0.0
    for key in labelCount:
        prob = float(labelCount[key])/numOfSample
        shanon_ent -= prob*log(prob, 2)

    return shanon_ent

def split_dataset(dataset, axis, value):
    """this will split dataset from axis, subset one whose feature
    `axis` equals `value`, another not equal, this function return subset one """
    ret_dataset = []
    for feat_vec in dataset:
        # extract feature `axis` from `dataset` and form `ret_dataset`
        if feat_vec[axis] == value:
            reduced_feats = feat_vec[:axis]
            reduced_feats.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feats)
    return ret_dataset

def find_best_splition(dataset):
    """ return int index of the best splition"""
    base_entrophy = calc_shanon_ent(dataset)
    num_feats = len(dataset[0]) - 1

    best_choice = { 'index': 0, 'entro': 0.0 }
    for i in range(num_feats):
        # find every possible value for each feature
        feat_list = [example[i] for example in dataset]
        unique_list = set(feat_list)
        # each value from a feature means a subtree
        # you have to calc the sum of each subtree's entrophy
        new_entro = 0.0
        for value in unique_list:
            sub_dataset = split_dataset(dataset, i, value)
            sub_data_entro = calc_shanon_ent(sub_dataset)
            prob = len(sub_dataset)/float(len(dataset))
            new_entro += prob * sub_data_entro
        if (base_entrophy - new_entro) > best_choice['entro']:
            best_choice['index'] = i
            best_choice['entro'] = base_entrophy - new_entro
    return best_choice['index'] # return a integer represent split axis

# def get_data(txt):
#     """ get data from txt, return dataste and label"""
#     dataset = np.genfromtxt('D:\study files\辅修计算机\机器学习\machine-learning\DecisionTree\\'+txt, delimiter='\t', dtype=np.str)
#     label = [last[-1] for last in data]
#     return dataset, label

def max_ent(class_label):
    return 0

def create_tree(dataset, labels):
    """ create a decision tree based on training set
    this tree is stored as dict structure in python"""
    # decide which class it is
    classLabel = [example[-1] for example in dataset]

    if classLabel.count(classLabel[0]) == 1:
        # all class in the dataset is the same, you can make a decision now
        return classLabel[0]
    if len(dataset[0]) == 1:
        # you don't have any other feature to tell the diffrence, you have to
        # make a better decision
        return max_ent(classLabel)
    # find a best split decision
    best_feat = find_best_splition(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label:{}}
    # how many feature value this `best_feat` has, you got how many branches
    feat_value = [example[best_feat] for example in dataset]
    unique_value = set(feat_value)
    del(labels[best_feat])
    # for a specific branch, do the recursive `create_tree`
    for value in unique_value:
        sub_labels = labels[:]
        subdata = split_dataset(dataset, best_feat, value)
        my_tree[best_feat_label][value] = create_tree(subdata, sub_labels)
    return my_tree

def get_right_lenses(trainedTree, testVec, labels):
    """ predict right lenses for patient"""
    first = trainedTree.keys()[0]
    secondDict = trainedTree[first]
    featIndex = labels.index(first)
    key = testVec[featIndex]
    value = secondDict[key]
    if isinstance(value, dict):
        class_label = get_right_lenses(value, testVec, labels)
    else:
        class_label = value
    return class_label

print(find_best_splition(create_dataset()[0]))
# data, label = get_data('lenses.txt')
dataset = np.genfromtxt('lenses.txt', delimiter='\t', dtype=np.str)
labels = ['age', 'eyesight', 'comfirm', 'condition', 'lenses']
ds_tree = create_tree(dataset.tolist(), labels)
test = ['young', 'hyper', 'no', 'reduced']
print(get_right_lenses(ds_tree, test, labels))
print(ds_tree)
