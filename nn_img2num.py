import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import matplotlib.pyplot as plt
import time
import numpy as np

class NnImg2Num():
    class net(nn.Module):
        def __init__(self):
            super(NnImg2Num.net,self).__init__()
            self.fc1 = nn.Linear(28*28,128)
            self.fc2 = nn.Linear(128,10)
            
        def forward(self,img):
            if len(img.shape)==3:
                x=img.view(img.shape[0],-1)
            elif len(img.shape)==2:
                x=img.view(1,-1)
            x1 = F.sigmoid(self.fc1(x))
            x2 = F.sigmoid(self.fc2(x1))
            return x2
        #return torch.argmax(x2,1)
    def __init__(self):
        self._inet=self.net()
        self._inet.__init__()
    
    def forward(self,img):
        scores=self._inet.forward(img)
        return torch.argmax(scores,1)
    def train(self):
        transform= transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

        trainset=torchvision.datasets.MNIST(
                root='./data',train=True,download=True,transform=transform)
        trainloader=torch.utils.data.DataLoader(
                trainset,batch_size=4,shuffle=True,num_workers=0)
        testset=torchvision.datasets.MNIST(
                root='./data',train=False,download=True,transform=transform)
        testloader=torch.utils.data.DataLoader(
                testset,batch_size=4,shuffle=False,num_workers=0)
        optimizer = optim.SGD(self._inet.parameters(),lr=0.01)
        
        eps=10
        eps_all=np.array(range(eps))+1
        loss_all=np.zeros((eps))
        time_all=np.zeros((eps))
        accuray_all=np.zeros((eps))
        time_start = time.clock()
        criterion = nn.MSELoss()
        
        
        for epochs in range(eps):
            running_loss =0.0
            train_num=0
            for batch_idx, (data, target) in enumerate(trainloader):
                x=data.view(data.shape[0],28,28)
                y=target.view(target.shape[0],1)
                optimizer.zero_grad()
                output=self._inet.forward(x)
                
                
                ytarget=torch.zeros((4,10))
                for i in range(len(y)):
                    ytarget[i,y[i]]=1
                    
                loss = criterion(output,ytarget)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_num+=data.size(0)
                
            #evaluate the accuracy on the test dataset
            correct = 0
            total = 0
            with torch.no_grad():
                for data, labels in testloader:
                    x=data.view(data.shape[0],28,28)
                    predicted=self.forward(x)
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
        plt.savefig('nn_loss.png')
        
        plt.figure(2)
        plt.plot(eps_all,accuray_all, linewidth=2.0)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy(%)')
        plt.savefig('nn_accuracy.png')
        