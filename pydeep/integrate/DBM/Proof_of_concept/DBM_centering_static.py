import numpy as numx
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.measuring as MEASURE
import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO

# Set the same seed value for all algorithms
numx.random.seed(42)

# Load Data
data = IO.load_MNIST("../../../../../data/mnist.pkl.gz",False)[0]

# Set dimensions
v11 = v12 = 28
v21 = v22 = 10
v31 = v32 = 10

N = v11 * v12
M = v21 * v22
O = v31 * v32

# Training parameters
batch_size = 100
epochs = 1
k_pos = 3
k_neg = 1

lr_W1 = 0.1
lr_W2 = 0.1

lr_b1 = 0.1
lr_b2 = 0.1
lr_b3 = 0.1

lr_o1 = 0.1
lr_o2 = 0.1
lr_o3 = 0.1

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

for epoch in range(0,epochs) :
    for b in range(0,data.shape[0],batch_size):

        # positive phase
        d1 = data[b:b+batch_size,:]
        id1 = numx.dot(d1-o1,W1)
        d3 = numx.zeros((batch_size,O))+o3
        for _ in range(k_pos):  
            d2 = Sigmoid.f( id1 + numx.dot(d3-o3,W2.T) + b2)
            #d2 = numx.float64(d2 > numx.random.random(d2.shape))
            d3 = Sigmoid.f(numx.dot(d2-o2,W2) + b3)
            #d3 = numx.float64(d3 > numx.random.random(d3.shape))

        # negative phase
        for _ in range(k_neg):  
            m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(m3-o3,W2.T) + b2)
            m2 = numx.float64(m2 > numx.random.random(m2.shape))
            m1 = Sigmoid.f(numx.dot(m2-o2,W1.T) + b1)
            m1 = numx.float64(m1 > numx.random.random(m1.shape))
            m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
            m3 = numx.float64(m3 > numx.random.random(m3.shape))

        # Estimate new means
        new_o1 = numx.mean(d1,axis=0)
        new_o2 = numx.mean(d2,axis=0)
        new_o3 = numx.mean(d3,axis=0)

        # Reparameterize
        b1 += lr_o2*numx.dot(new_o2-o2,W1.T)
        b2 += lr_o1*numx.dot(new_o1-o1,W1) 
        b2 += lr_o3*numx.dot(new_o3-o3,W2.T)
        b3 += lr_o2*numx.dot(new_o2-o2,W2)

        # Shift means
        o1 = (1.0-lr_o1)*o1 + lr_o1*new_o1
        o2 = (1.0-lr_o2)*o2 + lr_o2*new_o2
        o3 = (1.0-lr_o3)*o3 + lr_o3*new_o3
        
        # Calculate gradients
        dW1 = (numx.dot((d1-o1).T,d2-o2)-numx.dot((m1-o1).T,m2-o2))/batch_size
        dW2 = (numx.dot((d2-o2).T,d3-o3)-numx.dot((m2-o2).T,m3-o3))/batch_size
        db1 = numx.mean(d1-m1,axis = 0).reshape(1,N)
        db2 = numx.mean(d2-m2,axis = 0).reshape(1,M)
        db3 = numx.mean(d3-m3,axis = 0).reshape(1,O)

        # Update Model
        W1 += lr_W1*dW1
        W2 += lr_W2*dW2
        b1 += lr_b1*db1
        b2 += lr_b2*db2
        b3 += lr_b3*db3

        print numx.mean(numxExt.get_norms(W1)),'\t',numx.mean(numxExt.get_norms(W2)),'\t',
        print numx.mean(numxExt.get_norms(b1-numx.dot(o2,W1.T))),'\t',numx.mean(numxExt.get_norms(b2-numx.dot(o1,W1)-numx.dot(o3,W2.T))),'\t',
        print numx.mean(numxExt.get_norms(b3-numx.dot(o2,W2))),'\t',numx.mean(o1),'\t',numx.mean(o2),'\t',numx.mean(o3)
        
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


VIS.show()