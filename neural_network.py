import numpy as np
from indexedproperty import indexedproperty

class NeuralNetwork:
    def __init__(self,ins):
        self._W=[];
        self._Layers=len(ins)-1
        self._Layer_size=ins
        for i in range(self._Layers):
            rnd=np.random.randn(ins[i]+1,ins[i+1]);
            self._W.append((rnd)/np.sqrt(ins[i]))
    @indexedproperty
    def getLayer(self,layer):
        return self._W[layer]
    @getLayer.setter
    def getLayer(self, layer,Win):
        self._W[layer]=np.array(Win)
    
    def forward(self,input):
        inn=np.array(input,dtype=float)
        if len(np.shape(inn))==1:
            self._activation=[np.reshape(inn,[-1,1])]        
        elif len(np.shape(inn))==2:
            self._activation=[inn]
            
        cols=self._activation[0].shape[1]
        for l in range(self._Layers):
            self._activation.append(1/(1+np.exp(-
            self._W[l].T.dot(
                    np.concatenate((self._activation[l],np.ones((1,cols))),axis=0)
                    ))))
        return self._activation[self._Layers];
    
    def backward(self,target,loss='MSE'):
        #calculate the loss using softmax
        ntar=np.array(target,dtype=float)
        ntar=np.reshape(ntar,[self._Layer_size[-1],-1])
        if loss=='MSE':
            diff=2*(self._activation[self._Layers]-ntar)/ntar.shape[1]
        elif loss=='CE':
            exps=np.exp(self._activation[self._Layers])
            diff=(exps/np.sum(exps)-ntar)/ntar.shape[1]
            
        self._dW=[None]*self._Layers
        for l in range(self._Layers-1,-1,-1):            
            dlinear=diff*self._activation[l+1]*(1-self._activation[l+1])
            cols=self._activation[l].shape[1]
            self._dW[l]=np.dot(
                    np.concatenate((self._activation[l],np.ones((1,cols))),axis=0),
                    dlinear.T)
            diff=np.dot(self._W[l][:-1,:],dlinear)
    
    def updateParams(self,eta):
        for i in range(len(self._W)):
            reg=1e-7
            self._W[i]=self._W[i]-eta*self._dW[i]#-reg*self._W[i]