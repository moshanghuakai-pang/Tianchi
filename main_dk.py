import numpy as np
import dataProcess
import model
import torch
import random
import math
from torch.autograd import Variable

trainNum,testNum = [1000,1100,1300,1200],[360,400,700,500]
#qso:1362,galaxy:5230,star:442968,unknown:34287
BATCH = 40
EPOCH  = 100
LR = 0.01


def MyLoss(predict,label):
    loss = 0
    length = predict.shape[0]
    predict = predict.view(-1)
    label = label.view(-1).type(torch.cuda.FloatTensor)
    #print 'predict type:',type(predict.data),'label type',type(label.data)
    loss -= torch.matmul(torch.log(predict+0.00000000001),label)
    loss -= torch.matmul(torch.log(1-predict+0.000000000001),1-label)*0.2
    return loss/length

def Train(input,label,net,optimizer):
    net.train()
    predict = net(input)
    lossTrain = MyLoss(predict,label)
    optimizer.zero_grad()
    lossTrain.backward()
    optimizer.step()
    return lossTrain

def Test(input,label,net):
    net.eval()
    predict = net(input)
    lossTest = MyLoss(predict,label)
    predict = predict.cpu().data.view(predict.shape[0],predict.shape[-1])


    output = np.zeros((predict.shape[0],1))
    #print 'predict shape:',predict.shape
    temp = torch.zeros(predict.shape[0],4,1)
    predictMax = np.max(predict.numpy(),1).reshape(predict.shape[0],1).repeat(predict.shape[-1],1)
    pos = np.argwhere((predict.numpy()-predictMax)==0)
    index = np.unique(pos[:,0],return_index=True)[1]
    pos = pos[index]
    output = list(pos[:,1])
    np.save('./debug_data/output.npy',np.array(output))


    '''
    med = np.median(predict.numpy(),axis=1).reshape(predict.shape[0],1).repeat(predict.shape[-1],axis=1)
    med = torch.from_numpy(med).type(torch.FloatTensor)
    #print 'predict:',predict,'\nmed:',med
    output = torch.ge(predict-med,0).type(torch.FloatTensor)
    np.save('./debug_data/output.npy',output.numpy())
    distance = torch.pow(label.cpu().data.numpy()-output,2)
    #print 'distance:',distance
    trueMark = torch.eq(torch.sum(distance,1),0).type(torch.FloatTensor)
    #print 'output:',output,'\ntrueMark:',trueMark,'\nlabel:',label'''
    return output,lossTest


def main(trainNum,testNum,BATCH,LR1):
    lossTrain,lossTest = [],[]
    fileTrainNames,fileTestNames,labelsTrain,labelsTest = dataProcess.DataSample(trainNum,testNum)
    lengthTrain = len(fileTrainNames)
    trainOrder = np.arange(lengthTrain)    
    net = model.Mynn(BATCH)
    net.cuda()
    #net.load_state_dict(torch.load('./net_test.pkl'))

    for i in range(EPOCH):
        random.shuffle(trainOrder)
        j = 1
        lossTemp = []
        if not i%20: LR1 *= 0.1

        '''

        while j*BATCH <= lengthTrain:            
            fileBatchNames = fileTrainNames[trainOrder[(j-1)*BATCH:j*BATCH]]
            trainData = Variable(dataProcess.DataLoader(fileBatchNames)).cuda()
            labelTrain = Variable(torch.from_numpy(labelsTrain[trainOrder[(j-1)*BATCH:j*BATCH]])).cuda()

            optimizer = torch.optim.Adam(net.parameters(),lr=LR1)
            loss = Train(trainData,labelTrain,net,optimizer).cpu().data
            lossTemp.append(loss)
            j+=1

        #lossTrain.append(np.array(lossTemp).mean())'''
        
        testData = Variable(dataProcess.DataLoader(fileTestNames)).cuda()
        trueMark,lossTestTemp=Test(testData,Variable(torch.from_numpy(labelsTest)).cuda(),net)
        
        
        print '---------------------------EPOCH',i,'---------------------------------'
        print 'lossTrain',np.array(lossTemp).mean()

        

        print 'lossTest:',lossTestTemp

        qsoNum = float(trueMark[:testNum[0]].count(0))
        #print 'qsoNum,testNum:',qsoNum,testNum[0]
        qsoRecall = qsoNum/(testNum[0]+1)
        #print 'qsoRecall:',qsoRecall
        qsoPrecision = qsoNum/(trueMark.count(0)+1)
        qsoF1 = qsoRecall*qsoPrecision/(qsoRecall+qsoPrecision+0.0000001)

        galaxyNum = float(trueMark[testNum[0]:sum(testNum[:2])].count(1))
        galaxyRecall = galaxyNum/(testNum[1]+1)
        galaxyPrecision = galaxyNum/(trueMark.count(1)+1)
        galaxyF1 = galaxyRecall*galaxyPrecision/(galaxyRecall+galaxyPrecision+0.00000001)

        starNum = float(trueMark[sum(testNum[:2]):sum(testNum[:3])].count(2))
        starRecall = starNum/(testNum[2]+1)
        starPrecision = starNum/(trueMark.count(2)+1)
        starF1 = starRecall*starPrecision/(starRecall+starPrecision+0.0000001)

        unknownNum = float(trueMark[sum(testNum[:3]):sum(testNum[:4])].count(3))
        unknownRecall = unknownNum/(testNum[3]+1)
        unknownPrecision = unknownNum/(trueMark.count(3)+1)
        unknownF1 = unknownRecall*unknownPrecision/(unknownRecall+unknownPrecision+0.0000001)


        lossTest.append(lossTestTemp.cpu().data)
        print 'Num:',[qsoNum,galaxyNum,starNum,unknownNum]
        print 'Recall:',[qsoRecall,galaxyRecall,starRecall,unknownRecall]
        print 'Precision:',[qsoPrecision,galaxyPrecision,starPrecision,unknownPrecision]
        print 'F1:',[qsoF1,galaxyF1,starF1,unknownF1]

        '''
        qsoNum = trueMark[:testNum[0]].sum()/testNum[0]
        galaxyNum = trueMark[testNum[0]:sum(testNum[:2])].sum()/testNum[1]
        starNum = trueMark[sum(testNum[:2]):sum(testNum[:3])].sum()/testNum[2]
        unknownNum = trueMark[sum(testNum[:3]):].sum()/testNum[3]
        lossTest.append(lossTestTemp.cpu().data)
        print 'qso:',qsoNum,'\ngalaxy:',galaxyNum,'\nstar:',starNum,'\nunknown:',unknownNum
        '''
        torch.save(net.state_dict(),'net_test1.pkl')
if __name__ == '__main__':
    print 'start...'
    main(trainNum,testNum,BATCH,LR)
    print 'testNum:',testNum
