import numpy as np
import torch
from sklearn import preprocessing
import os
import xlrd as xl
import random


def Encoder(data):
    length = data.shape[0]
    label = np.zeros((length,4))
    for i in range(length):
        if data[i] == 'qso':label[i] = np.array([1,0,0,0])
        elif data[i] == 'galaxy':label[i] = np.array([0,1,0,0])
        elif data[i] == 'star':label[i] = np.array([0,0,1,0])
        elif data[i] == 'unknown':label[i] = np.array([0,0,0,1])
    return label


def DataSample(trainNum,testNum):
    fileList = os.listdir('./totaldata/dataIndexSeparate')
    #print 'fileList--->',fileList,'<----'
    fileTrain,fileTest = [],[]
    labelTrain,labelTest = [],[]
    for index,fName in enumerate(fileList):
        temp = []
        tempLabel = []
        data = xl.open_workbook('./totaldata/dataIndexSeparate/'+fName)
        table = data.sheets()[0]
        nrows = table.nrows
        for i in range(nrows):
            temp.append(str(int(table.row_values(i)[0])))
            #print 'str--->',table.row_values(i)[1]
            tempLabel.append(table.row_values(i)[1])
        #print 'tempLabel:',tempLabel[:10]
        temp = np.array(temp)
        tempLabel = np.array(tempLabel)
        fileIndex = np.arange(nrows)
        random.shuffle(fileIndex)
        #print 'Index--->',type(fileIndex)
        fileTrain.extend(list(temp[fileIndex[:trainNum[index]]]))
        fileTest.extend(list(temp[fileIndex[trainNum[index]:trainNum[index]+testNum[index]]]))
        #print 'Index--->',tempLabel[0]
        labelTrain.extend(list(Encoder(tempLabel[fileIndex[:trainNum[index]]])))
        labelTest.extend(list(Encoder(tempLabel[fileIndex[trainNum[index]:trainNum[index]+testNum[index]]])))
    return np.array(fileTrain),np.array(fileTest),np.array(labelTrain),np.array(labelTest)


def DataLoader(dataFile):
    length = len(dataFile)
    data = torch.zeros(length,1,50,52)
    temp = []
    for i in range(length):
        fr = open('./totaldata/datatrain/'+dataFile[i]+'.txt')
        temp = fr.readline().strip().split(',')
        temp = np.array([float(j) for j in temp])
        temp = preprocessing.scale(temp)
        temp = torch.from_numpy(temp).contiguous().view(1,50,52)
        #print 'temp shape--->',temp,'\ndata--->',data
        data[i] = temp
    return data


if __name__ == '__main__':
    '''
    path = '1000000'
    #print DataLoader(path)
    trainNum,testNum = [2,1,3,2],[2,2,2,2]
    fileTrain,fileTest,labelTrain,labelTest = DataSample(trainNum,testNum)
    print 'fileTrain--->',fileTrain,'\nfileTest--->',fileTest,'\nlabelTrain',labelTrain,'\nlabelTest',labelTest
    print '-data-->',DataLoader(fileTrain)'''
    
