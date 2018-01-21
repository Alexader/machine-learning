import numpy as np
import codecs
import bayes
baseURI = 'D:\study files\辅修计算机\机器学习\machine-learning\Bayes\\'

def bagOfwords2Vec(input_set, input_words):
    ''' record word frequency'''
    retVec = np.zeros(len(input_set))
    for word in input_words:
      if word in input_set:
        retVec[input_set.index(word)] += 1
    return retVec

def textParse(txt):
    ''' parse long email text into list of words'''
    import re
    listOfTokens = re.split(r'\w*', txt)
    return [token.lower() for token in listOfTokens if len(token) > 1]

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        # assume we have 25 emails for normal email and spam
        wordList = textParse(codecs.open(baseURI + 'email/spam/%d.txt' % i, encoding='ANSI').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open(baseURI + 'email/ham/%d.txt' % i, encoding='ANSI').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.create_vocab_list(docList)#create vocabulary
    trainingSet = list(range(50)); testSet=[]           #create test set
    # this will pop out 10 emails of traning set for testing algorithm randomly
    for i in range(10):
        # np.random.uniform(low, high) draw samples from a uniform distribution
        # The probability density function of the uniform distribution is p(x)=1/(high-low)
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:# train the classifier (get probs) trainNB0
        trainMat.append(bagOfwords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = bayes.trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfwords2Vec(vocabList, docList[docIndex])
        if bayes.classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText

spamTest()
