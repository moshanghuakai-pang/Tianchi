import torch
import torch.nn as nn

def Cov(inchannel,outchannel,size,stri,pad):
    return nn.Conv2d(inchannel,outchannel,size,stride=stri,padding = pad)

class Mynn(nn.Module):
    def __init__(self,batch):
        super(Mynn,self).__init__()

        self.cov1 = Cov(1,10,5,1,2)
        self.re1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(3,stride=2,padding=1)

        self.cov2 = Cov(10,10,5,1,2)
        self.re2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(3,stride=2,padding=1)

        self.cov3 = Cov(10,15,3,2,1)
        self.re3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(3,stride=2,padding=0)

        self.f = nn.Linear(135,4)

    def forward(self,x):
        out = self.cov1(x)
        #print 'x.shape:',x.shape,'out_cov1:',out.shape
        out = self.re1(out)
        out = self.maxpool1(out)
        #print out.shape
        out = self.cov2(out)
        out = self.re2(out)
        out = self.maxpool2(out)

        out = self.cov3(out)
        out = self.re3(out)
        out = self.maxpool3(out)

        out = out.view(x.shape[0],1,135)
        out = self.f(out)
        out = out.view(out.shape[0],out.shape[-1])
        #print 'befor softmax:',out.shape

        return nn.Softmax()(out)
