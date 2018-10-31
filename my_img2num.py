import numpy as np
from neural_network import NeuralNetwork
import torchvision
import torchvision.transforms as transforms
import torch

import matplotlib.pyplot as plt
import time

class MyImg2Num(NeuralNetwork):
    def __init__(self):
        self.layersizes=(784,128,10)
        NeuralNetwork.__init__(self,self.layersizes)
    def forward(self,img):
        if len(img.shape)==3:
            img2=np.transpose(img,(1,2,0))
            img2=np.reshape(img2,(-1,img2.shape[2]))
        elif len(img.shape)==2:
            img2=np.reshape(img,(-1,1))
        
        output=NeuralNetwork.forward(self,img2)
        return np.argmax(output,axis=0)
        
    def train(self):
        transform= transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

        trainset=torchvision.datasets.MNIST(
                root='../data',train=True,download=True,transform=transform)
        trainloader=torch.utils.data.DataLoader(
                trainset,batch_size=4,shuffle=True,num_workers=0)
        testset=torchvision.datasets.MNIST(
                root='./data',train=False,download=True,transform=transform)
        testloader=torch.utils.data.DataLoader(
                testset,batch_size=4,shuffle=False,num_workers=0)
                
        eps=10
        eps_all=np.array(range(eps))+1
        loss_all=np.zeros((eps))
        time_all=np.zeros((eps))
        time_start = time.clock()
        accuray_all=np.zeros((eps))
        for epochs in range(eps):
            running_loss=0.0
            train_num=0
            for batch_idx, (data, target) in enumerate(trainloader):
                xa=np.array(data)
                xa=np.reshape(xa,(xa.shape[0],28,28))
                train_num+=xa.shape[0]
                
                ya=np.array(target)
                ya=np.reshape(ya,(ya.shape[0],1))
                
                xaf=np.transpose(xa,(1,2,0))
                xaf=np.reshape(xaf,(-1,xaf.shape[2]))
                yt=np.zeros((10,4))
                for i in range(len(ya)):
                    yt[ya[i],i]=1
                
                output=NeuralNetwork.forward(self,xaf)
                method='MSE'
                self.backward(yt,method)
                self.updateParams(0.001)
                if method=='CE':
                    expo=np.exp(output)
                    loss=np.mean(-output[ya]+np.log(np.sum(expo)))
                elif method=='MSE':
                    loss=np.mean(np.square(output-yt))
                running_loss +=loss
                
                
            correct = 0
            total = 0
            for data, labels in testloader:
                xa=np.array(data)
                xa=np.reshape(xa,(xa.shape[0],28,28))
                predicted=self.forward(xa)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()   
            
            elapsed_time=(time.clock() - time_start)
            running_loss/=train_num
            accuracy=correct/total*100
            print('Epochs:%d, Loss:%.3e, time:%.1f, accuracy:%d%%'%
                  (epochs+1,running_loss,elapsed_time,accuracy))  
            loss_all[epochs]=running_loss
            time_all[epochs]=elapsed_time
            accuray_all[epochs]=accuracy
            
            
        plt.figure(1)
        plt.plot(eps_all,loss_all, linewidth=2.0)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('my_loss.png')
        
        plt.figure(2)
        plt.plot(eps_all,accuray_all, linewidth=2.0)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy(%)')
        plt.savefig('nn_accuracy.png')