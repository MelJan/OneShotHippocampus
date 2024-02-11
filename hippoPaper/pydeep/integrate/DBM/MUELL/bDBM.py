import numpy as numx
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.measuring as MEASURE
import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO

# Set the same seed value for all algorithms
numx.random.seed(42)

# Load Data
data = IO.load_MNIST("../../../workspacePy/data/mnist.pkl.gz",False)[0]

# Set dimensions
v11 = v12 = 28
v21 = v22 = 16
v31 = v32 = 25

N = v11 * v12
M = v21 * v22
O = v31 * v32

# Training parameters
batch_size = 25
epochs = 2
k_pos = 3
k_neg = 1

lr_W1 = 0.05
lr_W2 = 0.05

lr_b1 = 0.05
lr_b2 = 0.05
lr_b3 = 0.05

lr_o1 = 0.0
lr_o2 = 0.001
lr_o3 = 0.001

# Initialize parameters
W1 = numx.random.randn(N, M) * 0.01
W2 = numx.random.randn(M, O) * 0.01

o1 = numx.mean(data, axis = 0).reshape(1,N)
o2 = numx.zeros((1,M)) + 0.5
o3 = numx.zeros((1,O)) + 0.5

b1 = Sigmoid.g(numx.clip(o1,0.001,0.999))
b2 = Sigmoid.g(numx.clip(o2,0.001,0.999))
b3 = Sigmoid.g(numx.clip(o3,0.001,0.999))

# Initialize negative Markov chain
m1 = o1+numx.zeros((batch_size,N))
m2 = o2+numx.zeros((batch_size,M))
m3 = o3+numx.zeros((batch_size,O))

# Start time measure and training
measurer = MEASURE.Stopwatch()

for epoch in range(0,epochs) :
    for b in range(0,data.shape[0],batch_size):
        
        print b/numx.float64(data.shape[0])

        #positive phase
        d1 = data[b:b+batch_size,:]
        id1 = numx.dot(d1-o1,W1)
        d3 = o3
        for _ in range(k_pos):  
            d2 = Sigmoid.f( id1 + numx.dot(d3-o3,W2.T) + b2)
            d2 = numx.float64(d2 > numx.random.random(d2.shape))
            d3 = Sigmoid.f(numx.dot(d2-o2,W2) + b3)
            d3 = numx.float64(d3 > numx.random.random(d3.shape))
        #negative phase
        for _ in range(k_neg):  
            m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(m3-o3,W2.T) + b2)
            m2 = numx.float64(m2 > numx.random.random(m2.shape))
            m1 = Sigmoid.f(numx.dot(m2-o2,W1.T) + b1)
            m1 = numx.float64(m1 > numx.random.random(m1.shape))
            m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
            m3 = numx.float64(m3 > numx.random.random(m3.shape))
        
        # Estimate new means
        new_o1 = d1.mean(axis=0)
        new_o2 = d2.mean(axis=0)
        new_o3 = d3.mean(axis=0)

        # Reparameterize
        b1 += lr_o2*numx.dot(new_o2-o2,W1.T)
        b2 += lr_o1*numx.dot(new_o1-o1,W1) + lr_o3*numx.dot(new_o3-o3,W2.T)
        b3 += lr_o2*numx.dot(new_o2-o2,W2)
        
        # Shift means
        o1 = (1.0-lr_o1)*o1 + lr_o1*new_o1
        o2 = (1.0-lr_o2)*o2 + lr_o2*new_o2
        o3 = (1.0-lr_o3)*o3 + lr_o3*new_o3

        # Calculate gradients
        dW1 = (numx.dot((d1-o1).T,d2-o2)-numx.dot((m1-o1).T,m2-o2))
        dW2 = (numx.dot((d2-o2).T,d3-o3)-numx.dot((m2-o2).T,m3-o3))
        
        db1 = (numx.sum(d1-m1,axis = 0)).reshape(1,N)
        db2 = (numx.sum(d2-m2,axis = 0)).reshape(1,M)
        db3 = (numx.sum(d3-m3,axis = 0)).reshape(1,O)

        # Update Model
        W1 += lr_W1/batch_size*dW1
        W2 += lr_W2/batch_size*dW2
        
        b1 += lr_b1/batch_size*db1
        b2 += lr_b2/batch_size*db2
        b3 += lr_b3/batch_size*db3

    # Estimate Meanfield reconstruction 
    e2 = Sigmoid.f(numx.dot(data-o1,W1) + b2)
    e1 = Sigmoid.f(numx.dot(e2-o2,W1.T) + b1)
    e3 = Sigmoid.f(numx.dot(e2-o2,W2) + b3)
    e2 = Sigmoid.f(numx.dot(e1-o1,W1) + numx.dot(e3-o3,W2.T) + b2)
    e1 = Sigmoid.f(numx.dot(e2-o2,W1.T) + b1)
    
    # Plot Error and parameter norms, should be the same for both variants
    print numx.mean((e1-data)**2),
    print numx.mean(numxExt.get_norms(W1)),'\t',numx.mean(numxExt.get_norms(W2)),'\t',
    print numx.mean(numxExt.get_norms(b1+numx.dot(0.0-o2,W1.T))),'\t',
    print numx.mean(numxExt.get_norms(b2+numx.dot(0.0-o1,W1) + numx.dot(0.0-o3,W2.T))),'\t',
    print numx.mean(numxExt.get_norms(b3+numx.dot(0.0-o2,W2))),'\t',numx.mean(o1),'\t',numx.mean(o2),'\t',numx.mean(o3)

# End time measure
measurer.end()
print
print 'End-time: \t', measurer.get_end_time()
print 'Training time:\t', measurer.get_interval()

# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(W1, v11,v12, v21,v22, border_size = 1,normalized = False), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(W1,W2), v11,v12, v31,v32, border_size = 1,normalized = False), 'Weights 2')

# Show some sampling
VIS.imshow_matrix(VIS.tile_matrix_columns(d1, v11, v12, 1, batch_size, 1, False),'d')
m2 = Sigmoid.f(numx.dot(d1-o1,W1) + numx.dot(d3-o3,W2.T) + b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = Sigmoid.f(numx.dot(m2-o2,W1.T) + b1)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v11, v12, 1, batch_size, 1, False),'m 1')

m1 = numx.float64(m1 > numx.random.random(m1.shape))
m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
m3 = numx.float64(m3 > numx.random.random(m3.shape))
m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(d3-o3,W2.T) + b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = Sigmoid.f(numx.dot(m2-o2,W1.T) + b1)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v11, v12, 1, batch_size, 1, False),'m 2')

m1 = numx.float64(m1 > numx.random.random(m1.shape))
m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
m3 = numx.float64(m3 > numx.random.random(m3.shape))
m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(d3-o3,W2.T) + b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = Sigmoid.f(numx.dot(m2-o2,W1.T) + b1)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v11, v12, 1, batch_size, 1, False),'m 3')

m1 = numx.float64(m1 > numx.random.random(m1.shape))
m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
m3 = numx.float64(m3 > numx.random.random(m3.shape))
m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(d3-o3,W2.T) + b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = Sigmoid.f(numx.dot(m2-o2,W1.T) + b1)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v11, v12, 1, batch_size, 1, False),'m 4')

from sklearn.linear_model import LogisticRegression

train_data, train_label, valid_data, valid_label, test_data, test_label = IO.load_MNIST("../../../workspacePy/data/mnist.pkl.gz",False)

train_data2 = Sigmoid.f(numx.dot(train_data-o1,W1) + b2)
test_data2 = Sigmoid.f(numx.dot(test_data-o1,W1) + b2)
valid_data2 = Sigmoid.f(numx.dot(valid_data-o1,W1) + b2)

logReg = LogisticRegression(C=1000.0, penalty='l2', tol=0.01)
logReg.fit(train_data2, train_label)
print logReg.score(train_data2, train_label)
print logReg.score(valid_data2, valid_label)
print logReg.score(test_data2, test_label)


train_data2 = Sigmoid.f(numx.dot(train_data2-o2,W2) + b3)
test_data2 = Sigmoid.f(numx.dot(test_data2-o2,W2) + b3)
valid_data2 = Sigmoid.f(numx.dot(valid_data2-o2,W2) + b3)

logReg = LogisticRegression(C=1000.0, penalty='l2', tol=0.01)
logReg.fit(train_data2, train_label)
print logReg.score(train_data2, train_label)
print logReg.score(valid_data2, valid_label)
print logReg.score(test_data2, test_label)


VIS.show()