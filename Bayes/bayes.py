import numpy as np


def loadDataSet():
    """ load data for testing"""  
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
                 ['fuck', 'you', 'stupid', 'animals', 'go', 'to', 'hell'],
                 ['love', 'my', 'cute', 'animals', 'thank']]
    classVec = [0, 1, 0, 1, 0, 1, 1, 0]   #1 is abusive, 0 not
    return postingList, classVec

def create_vocab_list(dataset):
    ''' get a set of vocabulation database'''
    vocab = set([])
    for doc in dataset:
        vocab = vocab | set(doc)
    return list(vocab)

def vocab2Vec(input_set, input_list):
    ''' input_set is the labeled dataset, input_list is the doc you want
    to filter, you will get a vector which record what words have showed up
    in vocab dataset'''
    # you can get the same length vector no matter how long the input_list is
    retVec = [0] * len(input_set)
    for word in input_list:
        if word in input_set:
            retVec[input_set.index(word)] = 1
        else:
            print("this word '%s' not in my vocab list" % word)
    return retVec

def trainNB0(trainMatrix, class_labels):
    ''' bayes formula is p(c|x1,x2...xn)=p(x1,x2...xn|c)*p(x1,x2...xn)/p(c).
    each xi means a word in doc, `c` means the class of a doc. We assume each word is independent
    so we have to calc each p(xi|c) and mutiply them'''
    num_mat = len(trainMatrix)
    num_words = len(trainMatrix[0])
    # get the p(c) whcih is the probability of abusive email
    p_abusive = sum(class_labels)/float(len(class_labels))
    # 0 is non-abusive email, 1 is abusive
    p0Vec = p1Vec = np.ones(num_words)
    # we got their init value nonzero in case in posotion of denomintor
    p0total = p1total = 2.0
    # get p(xi|c) value
    for i in range(num_mat):
        if class_labels[i] == 1:
            # all these are vector add and multiply
            p0Vec += trainMatrix[i]
            p0total += sum(trainMatrix[i])
        if class_labels[i] == 0:
            p1Vec += trainMatrix[i]
            p1total += sum(trainMatrix[i])
    # cause these probability maybe way too small to cause overflow
    # we use log to deal with it
    p0 = np.log(p0Vec/p0total)
    p1 = np.log(p1Vec/p1total)
    return p0, p1, p_abusive

def classifyNB(vec2tclassify, p0vec, p1vec, p_class1):
    ''' `vec2tclassify` is the generated vector to be classified, p0vec is the probability for
    judging an email for non-abusive, p1vect is the opposite. p_class1 is the probability for
    abusive email showing up in database'''
    p0 = sum(vec2tclassify * p0vec) + np.log(1.0 - p_class1)
    p1 = sum(vec2tclassify * p1vec) + np.log(p_class1)
    print("p0: ", np.exp(p0))
    print("p1: ", np.exp(p1))
    if p0 > p1:
        return 0
    else:
        return 1

def test():
    posts, classlist = loadDataSet()
    vocab = create_vocab_list(posts)
    trainMatrix = []
    for doc in posts:
        trainMatrix.append(vocab2Vec(vocab, doc))
    p0, p1, p_abusive = trainNB0(np.array(trainMatrix), np.array(classlist))
    test0 = ['love', 'my', 'cute']
    print(test0, " classfy as: ", classifyNB(vocab2Vec(vocab, test0), p0, p1, p_abusive))
    test1 = ['fuck', 'hell']
    print(test1, " classfy as: ", classifyNB(vocab2Vec(vocab, test1), p0, p1, p_abusive))
if __name__ == '__main__':
    test()
