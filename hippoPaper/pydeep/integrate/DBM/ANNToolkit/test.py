import numpy as numx
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.measuring as MEASURE
import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO
from DBMModel import Weight_layer, Convolutional_weight_layer, Binary_layer

# Set the same seed value for all algorithms
numx.random.seed(42)

# Load Data
data, data_l, valid , valid_l , test, test_l= IO.load_MNIST("../../../../../data/mnist.pkl.gz",False)

# Set dimensions
v11 = v12 = 28
v21 = v22 = 15
v31 = v32 = 10
v41 = v42 = 5

N = v11 * v12
M = v21 * v22
O = v31 * v32
P = v41 * v42

wl1 = Convolutional_weight_layer(input_dim = N, 
                                 output_dim = M, 
                                 mask = Convolutional_weight_layer.construct_gauss_filter(3,3,variance = 0.5),
                                 initial_weights = 0.01,
                                 #connections = Weight_layer.generate_2D_connection_matrix(28, 28, 10, 10, 9, 9, False), 
                                 dtype = numx.float64)
wl2 = Convolutional_weight_layer(input_dim = M, 
                                 output_dim = O, 
                                 initial_weights = 0.01,
                                 mask = Convolutional_weight_layer.construct_gauss_filter(3,3,variance = 0.5),
                                 #connections = Weight_layer.generate_2D_connection_matrix(19, 19, 10, 10, 9, 9, False), 
                                 dtype = numx.float64)
wl3 = Convolutional_weight_layer(input_dim = O, 
                                 output_dim = P, 
                                 mask = Convolutional_weight_layer.construct_gauss_filter(3,3,variance = 0.5),
                                 initial_weights = 0.01,
                                 #connections = Weight_layer.generate_2D_connection_matrix(9, 9, 10, 10, 9, 9, False), 
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

l4 = Binary_layer(wl3, 
                  None, 
                  data = None, 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

# Initialize parameters
max_epochs = 10
batch_size = 100
k_pos = 3
k_neg = 1

lr_W1 = 0.01
lr_W2 = 0.01
lr_W3 = 0.01

lr_b1 = 0.01
lr_b2 = 0.01
lr_b3 = 0.01
lr_b4 = 0.01

lr_o1 = 0.1
lr_o2 = 0.1
lr_o3 = 0.1
lr_o4 = 0.1

# Initialize negative Markov chain
x_m = numx.zeros((batch_size,v11*v12))+l1.offset
y_m = numx.zeros((batch_size,v21*v22))+l2.offset
z_m = numx.zeros((batch_size,v31*v32))+l3.offset
a_m = numx.zeros((batch_size,v41*v42))+l4.offset

# Reparameterize RBM such that the inital setting is the same for centereing and centered training
l1.bias += numx.dot(0.0-l2.offset,wl1.weights.T)
l2.bias += numx.dot(0.0-l1.offset,wl1.weights) + numx.dot(0.0-l3.offset,wl2.weights.T)
l3.bias += numx.dot(0.0-l2.offset,wl2.weights) + numx.dot(0.0-l4.offset,wl3.weights.T)
l4.bias += numx.dot(0.0-l3.offset,wl3.weights)

# Start time measure and training
measurer = MEASURE.Stopwatch()

for epoch in range(0,max_epochs) :
    for b in range(0,data.shape[0],batch_size):
        
        #print b/numx.float64(data.shape[0])
        
        #positive phase
        x_d = data[b:b+batch_size,:]
        y_d = numx.zeros((batch_size,v21*v22))+l2.offset
        z_d = numx.zeros((batch_size,v31*v32))+l3.offset
        a_d = numx.zeros((batch_size,v41*v42))+l4.offset
        for i in range(k_pos):
            y_d = l2.activation(x_d, z_d)[0]
            #y_d = l2.sample(y_d)
            a_d = l4.activation(z_d, None)[0]
            #a_d = l4.sample(a_d)
            z_d = l3.activation(y_d, a_d)[0]
            #z_d = l3.sample(z_d)
            
        #negative phase
        for i in range(k_neg):
            y_m = l2.activation(x_m, z_m)
            y_m = l2.sample(y_m)
            a_m = l4.activation(z_m, None)
            a_m = l4.sample(a_m)
            x_m = l1.activation(None,y_m)
            x_m = l1.sample(x_m)  
            z_m = l3.activation(y_m, a_m)
            z_m = l3.sample(z_m) 

        # Estimate new means and update
        l1.update_offsets(numx.mean(x_d,axis = 0).reshape(1,x_d.shape[1]), lr_o1)
        l2.update_offsets(numx.mean(y_d,axis = 0).reshape(1,y_d.shape[1]), lr_o2)
        l3.update_offsets(numx.mean(z_d,axis = 0).reshape(1,z_d.shape[1]), lr_o3)
        l4.update_offsets(numx.mean(a_d,axis = 0).reshape(1,a_d.shape[1]), lr_o4)
        
        # Calculate centered weight gradients and update
        wl1_grad = wl1.calculate_weight_gradients(x_d, y_d, x_m, y_m, l1.offset, l2.offset)
        wl2_grad = wl2.calculate_weight_gradients(y_d, z_d, y_m, z_m, l2.offset, l3.offset)
        wl3_grad = wl3.calculate_weight_gradients(z_d, a_d, z_m, a_m, l3.offset, l4.offset)
        wl1.update_weights(lr_W1*wl1_grad, None, None)
        wl2.update_weights(lr_W2*wl2_grad, None, None)
        wl3.update_weights(lr_W3*wl3_grad, None, None)
        
        # Calsulate centered gradients for biases and update
        grad_b1 =l1.calculate_gradient_b(x_d, x_m, None, l2.offset, None, wl1_grad)
        grad_b2 =l2.calculate_gradient_b(y_d, y_m, l1.offset, l3.offset, wl1_grad, wl2_grad)
        grad_b3 =l3.calculate_gradient_b(z_d, z_m, l2.offset, l4.offset, wl2_grad, wl3_grad)
        grad_b4 =l4.calculate_gradient_b(a_d, a_m, l3.offset, None, wl3_grad, None)

        #Einbauen letzten zustande saven in Unit layer
        l1.update_biases(lr_b1*grad_b1)
        l2.update_biases(lr_b2*grad_b2)
        l3.update_biases(lr_b3*grad_b3)
        l4.update_biases(lr_b4*grad_b4)
    
    '''    
    # Estimate Meanfield reconstruction 
    x_e = data[0:1000]
    y_e = numx.zeros((1000,v21*v22))+l2.offset
    z_e = numx.zeros((1000,v31*v32))+l3.offset
    a_e = numx.zeros((1000,v41*v42))+l4.offset
    for i in range(1):
        y_e = l2.activation(x_e, z_e)[0]
        #y_e = l2.sample(y_e)
        a_e = l4.activation(z_e, None)[0]
        #a_e = l4.sample(a_e)
        z_e = l3.activation(y_e, a_e)[0]
        #z_e = l3.sample(z_e)
        x_e = l1.activation(None, y_e)[0]
        #x_e = l1.sample(x_e)
    '''
    l1.bias *= 0.0 #+ numxExt.get_norms(l1.bias, None)
    l2.bias *= 0.0 #+ numxExt.get_norms(l2.bias, None)
    l3.bias *= 0.0 #+ numxExt.get_norms(l3.bias, None)
    l4.bias *= 0.0 #+ numxExt.get_norms(l4.bias, None)
    a_e = numx.eye(P,P)
    z_e = numx.zeros((P,O))
    y_e = numx.zeros((P,M))
    x_e = numx.zeros((P,N))
    for i in range(0):
        z_e = l3.activation(y_e, a_e)[0]
        #z_e = l3.sample(z_e)
        y_e = l2.activation(x_e, z_e)[0]
        #y_e = l2.sample(y_e)
        x_e = l1.activation(None, y_e)[0]
        #x_e = l1.sample(x_e)

    z_e = l3.activation(y_e, a_e)[0]
    for i in range(P):
        val = numx.max(z_e[i])
        pos = numx.argmax(z_e[i])
        z_e[i] *= 0.0
        z_e[i,pos] += val
    #z_e = l3.sample(z_e)
    y_e = l2.activation(x_e, z_e)[0]
    for i in range(P):
        val = numx.max(y_e[i])
        pos = numx.argmax(y_e[i])
        y_e[i] *= 0.0
        y_e[i,pos] += val
    #y_e = l2.sample(y_e)
    x_e = l1.activation(None, y_e)[0]
    
    # Plot Error and parameter norms, should be the same for both variants
    print numx.mean((x_e-data[0:P])**2),'\t',numx.mean(numxExt.get_norms(wl1.weights)),'\t',numx.mean(numxExt.get_norms(wl2.weights)),'\t',numx.mean(numxExt.get_norms(wl3.weights))
    #print numx.mean(numxExt.get_norms(wl2.weights)),'\t',numx.mean(numxExt.get_norms(l1.bias)),'\t',
    #print numx.mean(numxExt.get_norms(l2.bias)),'\t', numx.mean(numxExt.get_norms(l3.bias)),'\t',
    #print numx.mean(l1.offset),'\t',numx.mean(l2.offset),'\t',numx.mean(l3.offset)


# End time measure
measurer.end()
print
print 'End-time: \t', measurer.get_end_time()
print 'Training time:\t', measurer.get_interval()
VIS.imshow_matrix(VIS.tile_matrix_rows(data[0:25].T, v11,v12, 10,10, border_size = 1,normalized = False), 'data') 
VIS.imshow_matrix(VIS.tile_matrix_rows(x_e[0:25].T, v11,v12, 10,10, border_size = 1,normalized = False), 'sample 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(y_e[0:25].T, v21,v22, 10,10, border_size = 1,normalized = False), 'sample 2')
VIS.imshow_matrix(VIS.tile_matrix_rows(z_e[0:25].T, v31,v32, 10,10, border_size = 1,normalized = False), 'sample 3')
VIS.imshow_matrix(VIS.tile_matrix_rows(a_e[0:25].T, v41,v42, 10,10, border_size = 1,normalized = False), 'sample 4')
# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(wl1.weights, v11,v12, v21,v22, border_size = 1,normalized = False), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(wl1.weights,wl2.weights), v11,v12, v31,v32, border_size = 1,normalized = False), 'Weights 2')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(wl1.weights,numx.dot(wl2.weights,wl3.weights)), v11,v12, v41,v42, border_size = 1,normalized = False), 'Weights 3')
VIS.imshow_matrix(VIS.tile_matrix_rows(wl3.weights, v31,v32, v41,v42, border_size = 1,normalized = False), 'Weights 4')
