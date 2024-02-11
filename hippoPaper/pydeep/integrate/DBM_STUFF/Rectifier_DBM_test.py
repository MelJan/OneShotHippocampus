import numpy as numx
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO
import pydeep.base.numpyextension as npExt
from scipy.signal import convolve2d
from pydeep.dbm.unit_layer import *
from pydeep.dbm.weight_layer import *
from pydeep.dbm.model import *

'''
def f(x, s = numx.linspace(0.0,10000.0,10000)):
    y = numx.copy(x)
    for i in range(x.shape[0]):
        temp = 0.0
        for u in range(s.shape[0]):
            temp += Sigmoid.f(x[i]-s[u]+0.5)
        y[i] =  temp
    return y
VIS.plot(numx.linspace(-10.0,10.0,200),numx.log(1+numx.exp(0.2*10*numx.linspace(-10.0,10.0,200))))
VIS.show()
exit()
'''
# Set the same seed value for all algorithms
numx.random.seed(42)

import time
localtime = time.localtime(time.time())
print "Local current time :", localtime

# Load Data
data = IO.load_mnist("../../../../data/mnist.pkl.gz",True)
label = npExt.get_binary_label(data[1])
data = data[0]

# Set dimensions
v11 = v12 = 28
v21 = v22 = 10
v31 = v32 = 10
v41 = 2
v42 = 5

N = v11 * v12
M = v21 * v22
O = v31 * v32
P = v41 * v42
'''
model = IO.load_object('dbm',True,False)
l1 = model.layers[0]
l2 = model.layers[1]
l3 = model.layers[2]
l4 = model.layers[3]

wl1 = l2.input_weight_layer
wl2 = l3.input_weight_layer
wl3 = l4.input_weight_layer
'''
wl1 = Weight_layer(input_dim = N, 
                   output_dim = M, 
                   initial_weights = 0.01,
                   dtype = numx.float64)
wl2 = Weight_layer(input_dim = M, 
                   output_dim = O, 
                   initial_weights = 0.01,
                   dtype = numx.float64)

wl3 = Weight_layer(input_dim = O,
                   output_dim = P,
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
                  wl3,
                  data = None, 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

l4 = Softmax_layer(wl3,
                  None,
                  data = None,
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

# Reparameterize RBM such that the inital setting is the same for centereing and centered training
l1.bias += numx.dot(0.0-l2.offset,wl1.weights.T)
l2.bias += numx.dot(0.0-l1.offset,wl1.weights) + numx.dot(0.0-l3.offset,wl2.weights.T)
l3.bias += numx.dot(0.0-l2.offset,wl2.weights) + numx.dot(0.0-l4.offset,wl3.weights.T)
l4.bias += numx.dot(0.0-l3.offset,wl3.weights)


# Initialize parameters
max_epochs = 10
batch_size = 10
k_d = 3
k_m = 3

lr_W = 0.01


lr_b = 0.01


lr_o = 0.01


# Initialize negative Markov chain
x_m = numx.zeros((batch_size,N))+l1.offset
y_m = numx.zeros((batch_size,M))+l2.offset
z_m = numx.zeros((batch_size,O))+l3.offset
a_m = numx.zeros((batch_size,P))+l4.offset
chain_m = [x_m,y_m,z_m,a_m]



print numx.mean(numxExt.get_norms(wl1.weights)), '\t', numx.mean(numxExt.get_norms(wl2.weights)), '\t', numx.mean(
    numxExt.get_norms(wl3.weights)), '\t',
print numx.mean(numxExt.get_norms(l1.bias)), '\t', numx.mean(numxExt.get_norms(l2.bias)), '\t',
print numx.mean(numxExt.get_norms(l3.bias)), numx.mean(numxExt.get_norms(l4.bias)), '\t',
print numx.mean(l1.offset), '\t', numx.mean(l2.offset), '\t', numx.mean(l3.offset), '\t', numx.mean(l4.offset)



# Start time measure and training
#measurer = MEASURE.Stopwatch()
model = DBM_model([l1,l2,l3,l4])

for epoch in range(0,max_epochs) :
    for b in range(0,data.shape[0],batch_size):
        
        x_d = data[b:b+batch_size,:]
        y_d = numx.zeros((batch_size,M))+l2.offset
        z_d = numx.zeros((batch_size,O))+l3.offset
        #a_d = numx.zeros((batch_size,P))+l4.offset
        a_d = label[b:b+batch_size,:]
        chain_d = [x_d,y_d,z_d,a_d]

        model.sample(chain_d, k_d, [True,False,False,True], True)
        
        # CD PCD Hybrid
        #chain_m[0] = x_d
        #model.sample(chain_m, k_m, [False,False,False], True)
        
        # CD 
        #chain_m[0] = numx.copy(x_d)
        #chain_m = model.sample(chain_d, k_m, [False,False,False,False], False)
        
        # PCD
        model.sample(chain_m, k_m, [False,False,False,False], True)
        
        model.update(chain_d, chain_m, lr_W, lr_b, lr_o)

    print numx.mean(numxExt.get_norms(wl1.weights,axis = None)),'\t',numx.mean(numxExt.get_norms(wl2.weights,axis = None)),'\t',numx.mean(numxExt.get_norms(wl3.weights,axis = None)),'\t',
    print numx.mean(numxExt.get_norms(l1.bias,axis = None)),'\t',numx.mean(numxExt.get_norms(l2.bias,axis = None)),'\t',
    print numx.mean(numxExt.get_norms(l3.bias,axis = None)),numx.mean(numxExt.get_norms(l4.bias,axis = None)),'\t',
    print numx.mean(l1.offset),'\t',numx.mean(l2.offset),'\t',numx.mean(l3.offset),'\t',numx.mean(l4.offset)


IO.save_object(model,'dbm_rec',True,False)

# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(wl1.weights, v11,v12, v21,v22, border_size = 1,normalized = True), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(wl1.weights,wl2.weights), v11,v12, v31,v32, border_size = 1,normalized = True), 'Weights 2')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(wl1.weights,numx.dot(wl2.weights,wl3.weights)), v11,v12, v41,v42, border_size = 1,normalized = True), 'Weights 3')

numx.random.seed(42)
# Show some sampling
x_d = data[0:batch_size, :]
y_d = numx.zeros((batch_size, M)) + l2.offset
z_d = numx.zeros((batch_size, O)) + l3.offset
# a_d = numx.zeros((batch_size,P))+l4.offset
a_d = label[0:batch_size, :]
chain_d = [x_d, y_d, z_d, a_d]
VIS.imshow_matrix(VIS.tile_matrix_columns(chain_d[0], v11, v12, 1, batch_size, 1, False),'data')
VIS.imshow_matrix(VIS.tile_matrix_columns(chain_d[3], v41*v42, 1, 1, batch_size, 1, False),'label')
chain_m = model.sample(chain_d, 100, [True,False,False,True], False)
VIS.imshow_matrix(VIS.tile_matrix_columns(l1.activation(None,chain_m[1])[0], v11, v12, 1, batch_size, 1, False),'data: data+label fix sample 100 step')
VIS.imshow_matrix(VIS.tile_matrix_columns(l4.activation(chain_m[2],None)[0], v41*v42, 1, 1, batch_size, 1, False),'label: data+label fix sample 100 step')
#VIS.imshow_matrix(VIS.tile_matrix_columns(chain_m[1], v21, v22, 1, batch_size, 1, False),'data 1 b')
#VIS.imshow_matrix(VIS.tile_matrix_columns(chain_m[2], v31, v32, 1, batch_size, 1, False),'data 2 b')
chain_m = model.sample(chain_m, 1, [False,True,True,False], False)
VIS.imshow_matrix(VIS.tile_matrix_columns(l1.activation(None,chain_m[1])[0], v11, v12, 1, batch_size, 1, False),'data: data+label fix sample 100 step')
VIS.imshow_matrix(VIS.tile_matrix_columns(l4.activation(chain_m[2],None)[0], v41*v42, 1, 1, batch_size, 1, False),'label: data+label fix sample 100 step')
#VIS.imshow_matrix(VIS.tile_matrix_columns(chain_m[1], v21, v22, 1, batch_size, 1, False),'data 1 a')
#VIS.imshow_matrix(VIS.tile_matrix_columns(chain_m[2], v31, v32, 1, batch_size, 1, False),'data 2 a')

numx.random.seed(42)
# Show some sampling
x_d = data[0:batch_size, :]
y_d = numx.zeros((batch_size, M)) + l2.offset
z_d = numx.zeros((batch_size, O)) + l3.offset
# a_d = numx.zeros((batch_size,P))+l4.offset
a_d = label[0:batch_size, :]
chain_d = [x_d, y_d, z_d, a_d]
chain_m = model.sample(chain_d, 100, [False,False,False,True], False)
VIS.imshow_matrix(VIS.tile_matrix_columns(l1.activation(None,chain_m[1])[0], v11, v12,  1, batch_size, 1, False),'data: label fix sample 100 step')
VIS.imshow_matrix(VIS.tile_matrix_columns(l4.activation(chain_m[2],None)[0], v41*v42, 1, 1, batch_size, 1, False),'label: label fix sample 100 step')


numx.random.seed(42)
# Show some sampling
x_d = data[0:batch_size, :]
y_d = numx.zeros((batch_size, M)) + l2.offset
z_d = numx.zeros((batch_size, O)) + l3.offset
# a_d = numx.zeros((batch_size,P))+l4.offset
a_d = label[0:batch_size, :]
chain_d = [x_d, y_d, z_d, a_d]
chain_m = model.sample(chain_d, 100, [False,False,False,False], False)
VIS.imshow_matrix(VIS.tile_matrix_columns(l1.activation(None,chain_m[1])[0], v11, v12,  1, batch_size, 1, False),'data: free sample 100 step')
VIS.imshow_matrix(VIS.tile_matrix_columns(l4.activation(chain_m[2],None)[0], v41*v42, 1, 1, batch_size, 1, False),'label: freex sample 100 step')

'''
VIS.imshow_matrix(VIS.tile_matrix_columns(l1.activation(None,chain_m[1])[0], v11, v12, 1, batch_size, 1, True),'d '+str(10))
chain_m = model.sample(chain_d, 10, [False,False,False,True], True)
VIS.imshow_matrix(VIS.tile_matrix_columns(l1.activation(None,chain_m[1])[0], v11, v12, 1, batch_size, 1, True),'d '+str(100))
chain_m = model.sample(chain_d, 100, [False,False,False,True], True)
VIS.imshow_matrix(VIS.tile_matrix_columns(l1.activation(None,chain_m[1])[0], v11, v12, 1, batch_size, 1, True),'d '+str(20))
model.sample(chain_m, 100, [False,False,False,False], True)
VIS.imshow_matrix(VIS.tile_matrix_columns(l1.activation(None,chain_d[1])[0], v11, v12, 1, batch_size, 1, True),'m '+str(100))
model.sample(chain_d, 100, [False,False,False,True], True)
VIS.imshow_matrix(VIS.tile_matrix_columns(l1.activation(None,chain_d[1])[0], v11, v12, 1, batch_size, 1, True),'m clamped'+str(300))
'''

VIS.show()