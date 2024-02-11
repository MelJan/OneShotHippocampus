import numpy as numx
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.base.costfunction as Cost
import pydeep.misc.measuring as MEASURE
import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO

class SoftMax(object):
    ''' Soft Max function.
          
    '''
    
    def f(self, x):
        ''' Calculates the SoftMax function value 
            for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the SoftMax function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        return numx.exp(x-numxExt.log_sum_exp(x, axis = 1).reshape(x.shape[0],1))
    
    def df(self, x):
        ''' Calculates the derivative of the SoftMax 
            function value for a given input x.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
                  
            :Parameters:
                y: All other values connected to this unit.
                  -type: scalar or numpy array
            
        :Returns:
            Value of the SoftMax function for x.
           -type: scalar or numpy array with the same shape as x.
              
        '''
        result = x[0]*numx.eye(x.shape[1],x.shape[1])-numx.dot(x[0].reshape(x.shape[1],1),x[0].reshape(1,x.shape[1])).reshape(1,x.shape[1],x.shape[1])
        for i in range(1,x.shape[0],1):
            result = numx.vstack((result,x[i]*numx.eye(x.shape[1],x.shape[1])-numx.dot(x[i].reshape(x.shape[1],1),x[i].reshape(1,x.shape[1])).reshape(1,x.shape[1],x.shape[1]))) 
        return result
        '''
        print numx.dot(h[1]*numx.eye(3,3)- numx.dot(h[1].reshape(3,1),h[1].reshape(1,3)),d[1].reshape(3,1))
        # pass a Nx(KxN) matrix
        result = x.reshape((1, 100*100))
        result = numx.tile(result, (100, 1))
        result_t = result.T
        result_t = numx.array_split(result_t, 100)
        result_t = numx.hstack(result_t)
        result *= (numx.tile( numx.eye(100), (1, 100)) - result_t)
        result *= numx.tile(y.reshape((1, 100*100)), (100, 1))
        result = numx.sum(result, axis=0)
        #return result.reshape((numberOfSamples, output_dim))
        '''
def int_to_bin(array):
    maxlabel = numx.max(array)+1
    result = numx.zeros((array.shape[0],maxlabel))
    for i in range(array.shape[0]):
        result[i,array[i]] = 1
    return result


class LogisticRegression2(object):
    
    def __init__(self, data, label):
        self.input_dim = data.shape[1]
        self.output_dim = label.shape[1]
        self.weights = numx.zeros((self.input_dim,self.output_dim))
        self.mean = numx.mean(data,axis = 0).reshape(1,self.input_dim)*0.0
        self.bias = numx.zeros((1,self.output_dim))
        self._temp_input = None
        self._temp_forward = None
        self._temp_backward = None
        
    def backwardPropagate(self, label):

        cost = Cost.NegLogLikelihood().df(self._temp_forward, label)
        result = None
        for i in range(self._temp_forward.shape[0]):
            '''   
            J = numx.zeros((x.shape[1],x.shape[1]))
            for u in range(x.shape[1]):
                for v in range(x.shape[1]):
                    if u == v:
                        J[u,v] = x[0,u]*(1.0-x[0,v])
                    else:
                        J[u,v] = -x[0,u]*x[0,v]
            '''
            x = self._temp_forward[i].reshape(1,self._temp_forward[i].shape[0])
            J1 = x*numx.eye(x.shape[1],x.shape[1])-numx.dot(x.T,x)
            temp = numx.dot(cost[i].reshape(1,cost.shape[1]),J1)

            if result == None:
                result = temp
            else:
                result = numx.vstack((result,temp))
        self._temp_backward = result
        check = self._temp_forward-label
        print numx.sum(numx.abs(result-check))
        exit()
        return self._temp_backward
    
    def forwardPropagate(self, x, update_mean):
        self._temp_input = x
        new_mean = numx.mean(self._temp_input,axis =0).reshape(1,self.input_dim)
        self.bias += update_mean*numx.dot(new_mean-self.mean,self.weights)
        self.mean = (1.0-update_mean)*self.mean + update_mean*new_mean
        act = numx.dot(self._temp_input-self.mean,self.weights)+self.bias
        #self._temp_forward = numx.exp(act-numxExt.log_sum_exp(act, axis = 1).reshape(act.shape[0],1))
        exp_x = numx.exp((act-(numx.max(act,axis = 1).reshape(act.shape[0],1))/2.0))
        self._temp_forward = exp_x/numx.sum(exp_x,axis = 1).reshape(act.shape[0],1)
        return self._temp_forward
    
    def _calculateGradient(self):
        gradW = numx.dot((self._temp_input-self.mean).T,self._temp_backward)/self._temp_input.shape[0]
        gradb = numx.dot(numx.ones((1,self._temp_input.shape[0])),self._temp_backward)/self._temp_input.shape[0]
        return gradW, gradb

class LogisticRegression(object):
    
    def __init__(self, data, label):
        self.input_dim = data.shape[1]
        self.output_dim = label.shape[1]
        self.weights = numx.zeros((self.input_dim,self.output_dim))
        self.mean = numx.mean(data,axis = 0).reshape(1,self.input_dim)*0.0
        self.bias = numx.zeros((1,self.output_dim))
        self._temp_input = None
        self._temp_forward = None
        self._temp_backward = None
        
    def backwardPropagate(self, label):
        self._temp_backward= self._temp_forward-label
        return self._temp_backward
    
    def forwardPropagate(self, x, update_mean):
        self._temp_input = x
        new_mean = numx.mean(self._temp_input,axis =0).reshape(1,self.input_dim)
        self.bias += update_mean*numx.dot(new_mean-self.mean,self.weights)
        self.mean = (1.0-update_mean)*self.mean + update_mean*new_mean
        act = numx.dot(self._temp_input-self.mean,self.weights)+self.bias
        #self._temp_forward = numx.exp(act-numxExt.log_sum_exp(act, axis = 1).reshape(act.shape[0],1))
        exp_x = numx.exp((act-(numx.max(act,axis = 1).reshape(act.shape[0],1))/2.0))
        self._temp_forward = exp_x/numx.sum(exp_x,axis = 1).reshape(act.shape[0],1)
        return self._temp_forward
    
    def _calculateGradient(self):
        gradW = numx.dot((self._temp_input-self.mean).T,self._temp_backward)/self._temp_input.shape[0]
        gradb = numx.dot(numx.ones((1,self._temp_input.shape[0])),self._temp_backward)/self._temp_input.shape[0]
        return gradW, gradb

numx.random.seed(42)


# Load Data
train, train_l, valid , valid_l , test, test_l= IO.load_MNIST("../../../PycharmProjects/data/mnist.pkl.gz",False)

train_l = int_to_bin(train_l)
test_l = int_to_bin(test_l)
valid_l = int_to_bin(valid_l)

batch_size = 100
W = numx.random.randn(784,10)*0.0
b = numx.zeros((1,10))

unit = SoftMax()
loss = Cost.CrossEntropyError()
for epoch in range(0,10) :
    for ba in range(0,train.shape[0],batch_size):
        batch = train[ba:ba+batch_size,:]
        lab = train_l[ba:ba+batch_size,:]
        act = unit.f(numx.dot(batch,W)+b)
        gradW = numx.dot(batch.T,act-lab)/batch.shape[0]
        gradb = numx.dot(numx.ones((1,batch.shape[0])),act-lab)/batch.shape[0]
        W -= 0.1*gradW
        b -= 0.1*gradb

    #print unit.f(numx.dot(batch,W)+b)
act = unit.f(numx.dot(train,W)+b)
    #pred = maxOut = numx.int32(act >= numx.atleast_2d(numx.max(act,axis = 1)).T)
    #print numx.mean(numx.sum(numx.abs(0.5*numx.abs(train_l-pred)),axis=1))
print numx.mean( numx.argmax(act,axis=1) <> numx.argmax(train_l,axis=1) )
print loss.f(act,train_l),'\t'

act = unit.f(numx.dot(valid,W)+b)
    #pred = maxOut = numx.int32(act >= numx.atleast_2d(numx.max(act,axis = 1)).T)
    #print numx.mean(numx.sum(numx.abs(0.5*numx.abs(valid_l-pred)),axis=1))
print numx.mean( numx.argmax(act,axis=1) <> numx.argmax(valid_l,axis=1) )
print loss.f(act,valid_l),'\t'
    
act = unit.f(numx.dot(test,W)+b)
    #pred = maxOut = numx.int32(act >= numx.atleast_2d(numx.max(act,axis = 1)).T)
    #print numx.mean(numx.sum(numx.abs(0.5*numx.abs(test_l-pred)),axis=1))
print numx.mean( numx.argmax(act,axis=1) <> numx.argmax(test_l,axis=1) )
print loss.f(act,test_l)
print
VIS.imshow_matrix(VIS.tile_matrix_rows(W, 28,28, 10,1, border_size = 1,normalized = True), 'Weights 1')     
numx.random.seed(42)

model = LogisticRegression(train,train_l)
for epoch in range(0,10) :
    for ba in range(0,train.shape[0],batch_size):
        batch = train[ba:ba+batch_size,:]
        lab = train_l[ba:ba+batch_size,:]
        model.forwardPropagate(batch, update_mean = 0)
        model.backwardPropagate(lab)
        gradW,gradb = model._calculateGradient()
        model.weights -= 0.1*gradW
        model.bias -= 0.1*gradb
        
#print unit.f(numx.dot(batch,W)+b)
act = model.forwardPropagate(train,0.0)
#pred = maxOut = numx.int32(act >= numx.atleast_2d(numx.max(act,axis = 1)).T)
#print numx.mean(numx.sum(numx.abs(0.5*numx.abs(train_l-pred)),axis=1))
print numx.mean( numx.argmax(act,axis=1) <> numx.argmax(train_l,axis=1) )
print loss.f(act,train_l),'\t'

act = model.forwardPropagate(valid,0.0)
#pred = maxOut = numx.int32(act >= numx.atleast_2d(numx.max(act,axis = 1)).T)
#print numx.mean(numx.sum(numx.abs(0.5*numx.abs(valid_l-pred)),axis=1))
print numx.mean( numx.argmax(act,axis=1) <> numx.argmax(valid_l,axis=1) )
print loss.f(act,valid_l),'\t'
    
act = model.forwardPropagate(test,0.0)
#pred = maxOut = numx.int32(act >= numx.atleast_2d(numx.max(act,axis = 1)).T)
#print numx.mean(numx.sum(numx.abs(0.5*numx.abs(test_l-pred)),axis=1))
print numx.mean( numx.argmax(act,axis=1) <> numx.argmax(test_l,axis=1) )
print loss.f(act,test_l)
print

print numx.sum(numx.abs(model.weights-W))
print numx.sum(numx.abs(model.bias-b))
VIS.imshow_matrix(VIS.tile_matrix_rows(model.weights, 28,28, 10,1, border_size = 1,normalized = True), 'Weights 2')

numx.random.seed(42)

model = LogisticRegression2(train,train_l)
for epoch in range(0,10) :
    for ba in range(0,train.shape[0],batch_size):
        batch = train[ba:ba+batch_size,:]
        lab = train_l[ba:ba+batch_size,:]
        model.forwardPropagate(batch, update_mean = 0)
        model.backwardPropagate(lab)
        gradW,gradb = model._calculateGradient()
        model.weights -= 0.1*gradW
        model.bias -= 0.1*gradb

#print unit.f(numx.dot(batch,W)+b)
act = model.forwardPropagate(train,0.0)
#pred = maxOut = numx.int32(act >= numx.atleast_2d(numx.max(act,axis = 1)).T)
#print numx.mean(numx.sum(numx.abs(0.5*numx.abs(train_l-pred)),axis=1))
print numx.mean( numx.argmax(act,axis=1) <> numx.argmax(train_l,axis=1) )
print loss.f(act,train_l),'\t'

act = model.forwardPropagate(valid,0.0)
#pred = maxOut = numx.int32(act >= numx.atleast_2d(numx.max(act,axis = 1)).T)
#print numx.mean(numx.sum(numx.abs(0.5*numx.abs(valid_l-pred)),axis=1))
print numx.mean( numx.argmax(act,axis=1) <> numx.argmax(valid_l,axis=1) )
print loss.f(act,valid_l),'\t'
    
act = model.forwardPropagate(test,0.0)
#pred = maxOut = numx.int32(act >= numx.atleast_2d(numx.max(act,axis = 1)).T)
#print numx.mean(numx.sum(numx.abs(0.5*numx.abs(test_l-pred)),axis=1))
print numx.mean( numx.argmax(act,axis=1) <> numx.argmax(test_l,axis=1) )
print loss.f(act,test_l)
print

print numx.sum(numx.abs(model.weights-W))
print numx.sum(numx.abs(model.bias-b))
VIS.imshow_matrix(VIS.tile_matrix_rows(model.weights, 28,28, 10,1, border_size = 1,normalized = True), 'Weights 3')
VIS.show()       


