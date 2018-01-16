# this will extract data from ouside txt file
# and do KNN algorithm


from numpy import *
import operator

def createData():
  # prepare data o ryou can import from txt file or so
  group = array([ [1.0,1.1],
                  [1.0,1.0],
                  [0,0],
                  [0,0.1] ])
  labels = ['A', 'A', 'B', 'B']
  return group, labels

def classfy(inX, data, labels, k):
  # inX is the input point, data is the preparing data, labels is
  # coresponding label for data, k is k-st data you want to choose 

  # calculating the distance between inX and every point in data
  dataColum = data.shape[1]
  dataRow = data.shape[0]     
  # tile(array, shape) will return an array with array repeated in shape
  # this part will calculate every distance from inX
  diffMat =  tile(inX, (dataRow, 1)) - data
  disMat = diffMat**2
  sqdistance = disMat.sum(axis=1)**0.5

  # you need to sort these distance and you have to keep track of their index
  # this is what argsort() do. It only returns sorted index without changing origin array
  sortedIndex = argsort(sqdistance)
  classfyLabel = {} # contain k-st label
  for i in range(k):
    voteLabel = labels[sortedIndex[i]]
    classfyLabel[voteLabel] = classfyLabel.get(voteLabel, 0) # dict.get(key, defaultReturnValue)
  return sorted(classfyLabel.items())[0][0]

def autoNorm(data):
  # each feature of record, its value varies a lot. It will have biasd info.
  # we assume all features are same important
  # normValue = (oldValue-min)/(max-min)
  min = data.min(0) # find max and min from axis `y`
  max = data.max(0)
  normMat = zeros(data.shape)
  normMat = (data-min)/(max-min)
  return normMat

def loadData():
  # this can load data from txt file about dating info and predict what kind of people do you expect
  # every record has four column, the first three column represent flight miles, video game time percentage
  # icecream assumption(KG). Forth column is label
  dataSet = loadtxt('datingTestSet2.txt', usecols = (0,1,2), dtype='float')
  labels = loadtxt('datingTestSet2.txt', usecols = (3,), dtype='S32')
  return dataSet, labels

def predictPerson(yourPerson):
  group, labels = loadData()
  print('Our system predict this person will be your %s type' % classfy(yourPerson, autoNorm(group), labels, 10))

group, labels = createData()
test = array([1,  0.8])
print(classfy(test, group, labels, 2))
group, labels = loadData()
test = array([4620,5.344,0.75952])
predictPerson(test)
