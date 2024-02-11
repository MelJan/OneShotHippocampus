import numpy as numx
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.measuring as MEASURE
import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO

# Set the same seed value for all algorithms
class CrossEntropyError(object):
    ''' Cross entropy functions.
          
    '''

    def f(self, x, t):
        ''' Calculates the CrossEntropy value for a given input x and target t.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
                  
                t: Target vales
                  -type: scalar or numpy array
            
        :Returns:
            Value of the CrossEntropy for x and t.
           -type: scalar or numpy array with the same shape as x and t.
              
        '''
        return -numx.mean(numx.sum(numx.log(x)*t+numx.log(1.0-x)*(1.0-t),axis = 1))
        
    def df(self, x, t):
        ''' Calculates the derivative of the CrossEntropy value for a 
            given input x and target t.
        
            :Parameters:
                x: Input data.
                  -type: scalar or numpy array
                  
                t: Target vales
                  -type: scalar or numpy array
            
        :Returns:
            Value of the derivative of the CrossEntropy for x and t.
           -type: scalar or numpy array with the same shape as x a d t.
              
        '''
        return -t/x+(1.0-t)/(1.0-x)


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
    
    def df(self, x, y):
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


def int_to_bin(array):
    maxlabel = numx.max(array)+1
    result = numx.zeros((array.shape[0],maxlabel))
    for i in range(array.shape[0]):
        result[i,array[i]] = 1
    return result

numx.random.seed(42)

# Load Data
train, train_l, valid , valid_l , test, test_l= IO.load_MNIST("../../../workspacePy/data/mnist.pkl.gz",False)

train_l = int_to_bin(train_l)
test_l = int_to_bin(test_l)
valid_l = int_to_bin(valid_l)

batch_size = 10
W = numx.random.randn(784,10)*0.0
b = numx.zeros((1,10))
unit = SoftMax()
loss = CrossEntropyError()
for epoch in range(0,73) :
    for b in range(0,train.shape[0],batch_size):
        batch = train[b:b+batch_size,:]
        lab = train_l[b:b+batch_size,:]
        act = unit.f(numx.dot(batch,W)+b)
        gradW = numx.dot(batch.T,lab-act)/batch.shape[0]
        gradb = numx.dot(numx.ones((1,batch.shape[0])),lab-act)/batch.shape[0]
        W += 0.01*gradW
        b += 0.01*gradb

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
VIS.show()       

