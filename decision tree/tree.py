from math import log
import operator as op
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
       # print(currentLabel)
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    #print(labelCounts)
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt -=prob*log(prob,2)
    #H(x) = E[I(xi)] = E[ log(2,1/p(xi)) ] = -∑p(xi)log(2,p(xi)) (i=1,2,..n)
    #信息熵的公式，可以参考学习
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def splitDataSet(dataSet,axis,value):
    '''此段代码应该这样理解，以数据的某一列作为特征值，来判断输入的数值的值
    和原训练集中符合的部分'''
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] == value:
                reducedFeatVec=featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numberFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numberFeatures):
        #下列两行创建唯一的分类标签列表
        #print(example[i]for example in dataSet)
        featList=[example[i]for example in dataSet]
        #print(example)
        print(featList)
        uniqueVals=set(featList)
        newEntropy=0.0
        #计算每种划分方式的信息熵
        for value in  uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            #计算最好的信息增益
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount+=1
    sortedClassCount=sorted(classCount.items(),key=op.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    #类别相同则停止划分
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #遍历完后返回所有特征时返回出现次数最多的类别
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    #得到列表包含的所有属性值
    del(labels[bestFeat])
    featValues=[example[bestFeat]for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
    
def classify(inputTree,featLabels,testVec):
    firstStrSide=list(inputTree.keys())
    firstStr=firstStrSide[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel
def showlabel(featLabels):
    print(featLabels)

def storeTree(inputTree,filename):
    import pickle as pick
    fw=open(filename,"wb+")
    pick.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle as pick
    fr=open(filename,'rb+')
    return pick.load(fr)
    
    
if __name__ =='__main__':
    myDat,labels=createDataSet()
    #labell = ['no surfacing','flippers']
    labelll=labels.copy()
    myTree=createTree(myDat,labels)
    #showlabel(labell)
    #showlabel(labels)
    storeTree(myTree,"classifyTree.txt")
    print(grabTree("classifyTree.txt"))
    #这两个方法方法在PY36中碰到了读取错误问题，应当注意PY36和PY27的文件读入输出区别
    #print(classify(myTree,labelll,[1,1]))
    #此处需要注意的是Python的引用问题，按照书上代码没有labels进行复制，这样子在classify时候的labels的内容是已经发生了改变了
    #不知道是py36和PY27的区别还是书上代码编写有误，反正在这里应该注意一下
        