import numpy as numx
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO
from scipy.signal import convolve2d
#from DBM_centered_flexible import Binary_layer
#from DBM.unit_layer import *
#from DBM.weight_layer import *

from pydeep.dbm.unit_layer  import *
from pydeep.dbm.weight_layer import *

# Set the same seed value for all algorithms
numx.random.seed(42)


# Load Data
data = IO.load_mnist("../../../../data/mnist.pkl.gz",False)[0]

# Set dimensions
v11 = v12 = 28
v21 = v22 = 10
v31 = v32 = 10

N = v11 * v12
M = v21 * v22
O = v31 * v32

wl1 = Weight_layer(input_dim = N, 
                   output_dim = M, 
                   initial_weights = 0.01,
                   dtype = numx.float64)
wl2 = Weight_layer(input_dim = M, 
                   output_dim = O, 
                   initial_weights = 0.01,
                   dtype = numx.float64)

l1 = Binary_layer(None, 
                  wl1, 
                  data = data, 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

l2 = Binary_layer(wl1, 
                  wl2, 
                  data = None, 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

l3 = Binary_layer(wl2, 
                  None, 
                  data = None, 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

# Initialize parameters
max_epochs = 1
batch_size = 100
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

# Initialize negative Markov chain
x_m = numx.zeros((batch_size,v11*v12))+l1.offset
y_m = numx.zeros((batch_size,v21*v22))+l2.offset
z_m = numx.zeros((batch_size,v31*v32))+l3.offset

# Reparameterize RBM such that the inital setting is the same for centereing and centered training
l1.bias += numx.dot(0.0-l2.offset,wl1.weights.T)
l2.bias += numx.dot(0.0-l1.offset,wl1.weights) + numx.dot(0.0-l3.offset,wl2.weights.T)
l3.bias += numx.dot(0.0-l2.offset,wl2.weights)

# Start time measure and training
#measurer = MEASURE.Stopwatch()

for epoch in range(0,max_epochs) :
    for b in range(0,data.shape[0],batch_size):
        
        #print b/numx.float64(data.shape[0])
   
        #positive phase
        x_d = data[b:b+batch_size,:]
        y_d = numx.zeros((batch_size,M))+l2.offset
        z_d = numx.zeros((batch_size,O))+l3.offset
        for _ in range(k_pos):
            y_d = l2.activation(x_d, z_d)
            y_d = l2.sample(y_d)
            z_d = l3.activation(y_d, None)
            z_d = l3.sample(z_d)

        #negative phase
        for _ in range(k_neg):
            y_m = l2.activation(x_m, z_m)
            y_m = l2.sample(y_m)
            x_m = l1.activation(None,y_m)
            x_m = l1.sample(x_m)  
            z_m = l3.activation(y_m, None)
            z_m = l3.sample(z_m) 
        
        # Estimate new means
        new_o1 = numx.mean(x_d,axis=0)
        new_o2 = numx.mean(y_d,axis=0)
        new_o3 = numx.mean(z_d,axis=0)

        l1.update_offsets(numx.mean(x_d,axis = 0).reshape(1,x_d.shape[1]), lr_o1)
        l2.update_offsets(numx.mean(y_d,axis = 0).reshape(1,y_d.shape[1]), lr_o2)
        l3.update_offsets(numx.mean(z_d,axis = 0).reshape(1,z_d.shape[1]), lr_o3)

        # Calculate normal gradients
        wl1_grad = wl1.calculate_weight_gradients(x_d, y_d, x_m, y_m, l1.offset, l2.offset)
        wl2_grad = wl2.calculate_weight_gradients(y_d, z_d, y_m, z_m, l2.offset, l3.offset)
        
        # Calsulate centered gradients for biases and update
        grad_b1 =l1.calculate_gradient_b(x_d, x_m, None, l2.offset, None, wl1_grad)
        grad_b2 =l2.calculate_gradient_b(y_d, y_m, l1.offset, l3.offset, wl1_grad, wl2_grad)
        grad_b3 =l3.calculate_gradient_b(z_d, z_m, l2.offset, None, wl2_grad, None)

        wl1.update_weights(lr_W1*wl1_grad, None, None)
        wl2.update_weights(lr_W2*wl2_grad, None, None)

        #Einbauen letzten zustande saven in Unit layer
        l1.update_biases(lr_b1*grad_b1, None, None)
        l2.update_biases(lr_b2*grad_b2, None, None)
        l3.update_biases(lr_b3*grad_b3, None, None)
        
        print numx.mean(numxExt.get_norms(wl1.weights)),'\t',numx.mean(numxExt.get_norms(wl2.weights)),'\t',
        print numx.mean(numxExt.get_norms(l1.bias)),'\t',numx.mean(numxExt.get_norms(l2.bias)),'\t',
        print numx.mean(numxExt.get_norms(l3.bias)),'\t',numx.mean(l1.offset),'\t',numx.mean(l2.offset),'\t',numx.mean(l3.offset)
        

# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(wl1.weights, v11,v12, v21,v22, border_size = 1,normalized = False), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(wl1.weights,wl2.weights), v11,v12, v31,v32, border_size = 1,normalized = False), 'Weights 2')

# Show some sampling
x_m = x_d
z_m = l3.offset
VIS.imshow_matrix(VIS.tile_matrix_columns(x_d, v11, v12, 1, batch_size, 1, False),'d')
for i in range(4):
    y_m = l2.activation(x_m, z_m)
    y_m = l2.sample(y_m)
    x_m = l1.activation(None,y_m)
    VIS.imshow_matrix(VIS.tile_matrix_columns(x_m[0], v11, v12, 1, batch_size, 1, False),'m '+str(i))
    x_m = l1.sample(x_m)  
    z_m = l3.activation(y_m, None)
    z_m = l3.sample(z_m) 
    
            
VIS.show()