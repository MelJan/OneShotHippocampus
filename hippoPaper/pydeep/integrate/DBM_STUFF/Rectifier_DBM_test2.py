import numpy as numx
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO
from scipy.signal import convolve2d
from pydeep.dbm.unit_layer import *
from pydeep.dbm.weight_layer import *
from pydeep.dbm.model import *

# Set the same seed value for all algorithms
numx.random.seed(42)

# Load Data
data = IO.load_MNIST("../../../../data/mnist.pkl.gz",False)[0]

# Set dimensions
v11 = v12 = 28
v21 = v22 = 5

N = v11 * v12
M = v21 * v22

wl1 = Weight_layer(input_dim = N, 
                   output_dim = M, 
                   initial_weights = 0.01,
                   dtype = numx.float64)

l1 = Binary_layer(None,
                  wl1, 
                  data = data, 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

l2 = Binary_layer(wl1,
                  None,
                  data = None, 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

# Initialize parameters
max_epochs = 1
batch_size = 100
k_d = 1
k_m = 20

lr_W = 0.1


lr_b = 0.1


lr_o = 0.1


# Initialize negative Markov chain
x_m = numx.zeros((batch_size,N))+l1.offset
y_m = numx.zeros((batch_size,M))+l2.offset
chain_m = [x_m,y_m]

# Reparameterize RBM such that the inital setting is the same for centereing and centered training
l1.bias += numx.dot(0.0-l2.offset,wl1.weights.T)
l2.bias += numx.dot(0.0-l1.offset,wl1.weights)

# Start time measure and training
#measurer = MEASURE.Stopwatch()
model = DBM_model([l1,l2])

for epoch in range(0,max_epochs) :
    for b in range(0,data.shape[0],batch_size):
        
        x_d = data[b:b+batch_size,:]
        y_d = numx.zeros((batch_size,M))+l2.offset
        chain_d = [x_d,y_d]
        model.sample(chain_d, k_d, [True,False], True)

        # CD PCD Hybrid
        #chain_m[0] = x_d
        #model.sample(chain_m, k_m, [False,False,False], True)
        
        # CD 
        #chain_m[0] = x_d
        #chain_m = model.sample(chain_d, k_m, [False,False,False], False)
        
        # PCD
        model.sample(chain_m, k_m, [False,False], True)
        
        model.update(chain_d, chain_m, lr_W, lr_b, lr_o)

    print numx.mean(numxExt.get_norms(wl1.weights)),'\t',
    print numx.mean(numxExt.get_norms(l1.bias)),'\t',numx.mean(numxExt.get_norms(l2.bias)),'\t',
    print numx.mean(l1.offset),'\t',numx.mean(l2.offset)
        

# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(wl1.weights, v11,v12, v21,v22, border_size = 1,normalized = False), 'Weights 1')

# Show some sampling
VIS.imshow_matrix(VIS.tile_matrix_columns(chain_d[0], v11, v12, 1, batch_size, 1, True),'m '+str(0))

model.sample(chain_d, 10, [False,False], True)
VIS.imshow_matrix(VIS.tile_matrix_columns(chain_d[0], v11, v12, 1, batch_size, 1, True),'m '+str(10))

model.sample(chain_d, 100, [False,True], True)
VIS.imshow_matrix(VIS.tile_matrix_columns(chain_d[0], v11, v12, 1, batch_size, 1, True),'m clamped'+str(10))
            
VIS.show()