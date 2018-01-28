import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(x))

def loadData():
    file = np.loadtxt('D:\\study files\\cslesson\\ML\\machine-learning\\Logistic\\testSet.txt', dtype=float, delimiter='\t')
    labels = []
    # set x_0 to 1.0 in this case
    data_mat = [np.append([1.0], example[:len(file[0])-1]) for example in file]
    for example in file:
        labels.append(int(example[-1]))
    return data_mat, labels

def gradient_decent(dataMat, class_labels):
    ''' dataMat is m*n matrix, class_labels is labels for each row
    find the most sufficient weight vector to make 
    '''
    data_matrix = np.mat(dataMat)
    class_matrix = np.mat(class_labels).transpose()

    alpha = 0.01 # step length
    max_iteration = 500
    m, n = np.shape(data_matrix)
    weights = np.ones((n, 1))
    for i in range(max_iteration):
        # do matrix multiply: m*1(h) = m*n multiply n*1
        h = sigmoid(data_matrix * weights)
        error = (class_matrix - h)
        # n*1 = n*1 - n*m multiply m*1
        weights = weights - alpha * data_matrix.transpose() * error
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadData()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, np.array(y)[0])
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

data, labels = loadData()
weights = gradient_decent(data, labels)
print(weights[2])
plotBestFit(weights)
