# -*- coding: utf8 -*-
# decision tree implementation

from math import log


def create_dataset():
    # create some data for test
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels

def calc_shanon_ent(dataSet):
    # formula for entrophy: Info(D) = -\sum_{i=1}^{m} p_i*log(p_i)
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
    # this will split dataset from axis, subset one whose feature
    # `axis` equals `value`, another not equal
    ret_dataset = []
    for feat_vec in dataset:
        # extract feature `axis` from `dataset` and form `ret_dataset`
        if feat_vec[axis] == value:
            reduced_feats = feat_vec[:axis]
            reduced_feats.extend(feat_vec[axis+1:])
            ret_dataset.append(reduced_feats)
    return ret_dataset

def find_best_splition(dataset):
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

print(find_best_splition(create_dataset()[0]))