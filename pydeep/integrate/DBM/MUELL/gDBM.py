import numpy as numx

import pydeep.base.numpyextension as numxExt
import pydeep.misc.io as IO
import pydeep.misc.measuring as MEASURE
import pydeep.misc.visualization as VIS
from pydeep.base.activationfunction import Sigmoid


def binary_unit(activation):
    return numx.float64(Sigmoid.f(activation) > numx.random.random(activation.shape))

def gaussian_unit(activation):
    return activation + numx.random.randn(activation.shape[0],activation.shape[1])


def generate_2D_connection_matrix(    input_x_dim, 
                                          input_y_dim, 
                                          field_x_dim, 
                                          field_y_dim, 
                                          overlap_x_dim, 
                                          overlap_y_dim, 
                                          wrap_around = True):
    if field_x_dim > input_x_dim:
        raise NotImplementedError("field_x_dim > input_x_dim is invalid!")
    if field_y_dim > input_y_dim:
        raise NotImplementedError("field_y_dim > input_y_dim is invalid!")
    if overlap_x_dim >= field_x_dim:
        raise NotImplementedError("overlap_x_dim >= field_x_dim is invalid!")
    if overlap_y_dim >= field_y_dim:
        raise NotImplementedError("overlap_y_dim >= field_y_dim is invalid!")
    
    matrix = None
    start_x = 0
    start_y = 0
    end_x = input_x_dim 
    end_y = input_y_dim 
    if wrap_around == False:
        end_x -= field_x_dim - 1
        end_y -= field_y_dim - 1
    step_x = field_x_dim - overlap_x_dim
    step_y = field_y_dim - overlap_y_dim
    
    for x in range(start_x,end_x,step_x):
        for y in range(start_y,end_y,step_y):
            column = numx.zeros((input_x_dim,input_y_dim))
            for i in range(x,x+field_x_dim,1):
                for j in range(y,y+field_y_dim,1):
                    column[i%input_x_dim,j%input_y_dim] = 1.0
            column = column.reshape((input_x_dim*input_y_dim))
            if matrix == None:
                matrix = column
            else:
                matrix = numx.vstack((matrix,column))
    return matrix.T

# Set the same seed value for all algorithms
numx.random.seed(42)

# Set dimensions
v11_org = v12_org = 14
v11 = v12 = 14
v21 = v22 = 14
v31 = v32 = 14

N_org = v11_org * v12_org
N = v11 * v12
M = v21 * v22
O = v31 * v32

conn1 = generate_2D_connection_matrix(input_x_dim = v11, 
                                     input_y_dim = v12, 
                                     field_x_dim = 8, 
                                     field_y_dim = 8, 
                                     overlap_x_dim = 7, 
                                     overlap_y_dim = 7, 
                                     wrap_around = True)

conn2 = generate_2D_connection_matrix(input_x_dim = v21, 
                                     input_y_dim = v22, 
                                     field_x_dim = 2, 
                                     field_y_dim = 2, 
                                     overlap_x_dim = 1, 
                                     overlap_y_dim = 1, 
                                     wrap_around = True)

# Load and whiten data
data = numx.random.permutation(IO.load_matlab_file('../../../workspacePy/data/NaturalImage.mat','rawImages'))
data = preprocessing.remove_rows_means(data)
pca = preprocessing.PCA(N_org, whiten= True)
pca.train(data)
data = pca.unproject(pca.project(data,12*12))
pca = preprocessing.ZCA(N_org)
pca.train(data)
data = pca.project(data)

# Training parameters
batch_size1 = 10
batch_size2 = 100
batch_size = batch_size1*batch_size2
epochs = 50
k_pos = 3
k_neg = 1

lr_W1 = 0.1
lr_W2 = 0.0

# Check if GRBM auf natural images trained, dann binary oben drauf lernt corners und  junktions
lr_b1 = 0.0
lr_b2 = 0.1
lr_b3 = 0.1

lr_o1 = 0.0
lr_o2 = 0.001
lr_o3 = 0.001

mom_W1 = 0.97
mom_W2 = 0.0

mom_b1 = 0.0
mom_b2 = 0.0
mom_b3 = 0.0

# Initialize parameters
W1 = numx.random.randn(N, M) * 0.01 
#W1 *= conn1
#W2 = numx.random.randn(M, O) * 0.01 
W2 = numx.ones((M,O))*conn2

o1 = numx.zeros((1,N))
o2 = numx.zeros((1,M))
o3 = numx.zeros((1,O))

b1 = numx.zeros((1,N))
b2 = numx.zeros((1,M))-4.0
b3 = numx.zeros((1,O))-4.0

dW1 = numx.zeros((N,M))
dW2 = numx.zeros((M,O))
        
db1 = numx.zeros((1,N))
db2 = numx.zeros((1,M))
db3 = numx.zeros((1,O))

# Initialize negative Markov chain
m1 = o1+numx.zeros((batch_size,N))
m2 = o2+numx.zeros((batch_size,M))
m3 = o3+numx.zeros((batch_size,O))

# Start time measure and training
measurer = MEASURE.Stopwatch()

for epoch in range(0,epochs) :
    data = numx.random.permutation(data)
    print epoch

    batch_size = batch_size1*batch_size2
    m1 = m1[0:batch_size,:]
    m2 = m1[0:batch_size,:]
    m3 = m3[0:batch_size,:]
    for b in range(0,data.shape[0],batch_size):
        
        #positive phase
        d1 = data[b:b+batch_size,:]
        id1 = numx.dot(d1-o1,W1)
        d3 = o3
        for _ in range(k_pos):  
            d2 = binary_unit(id1 + numx.dot(d3-o3,W2.T) + b2)
            d3 = binary_unit(numx.dot(d2-o2,W2) + b3)
            #d2 = Sigmoid.f( id1 + numx.dot(d3-o3,W2.T) + b2)
            #d2 = numx.float64(d2 > numx.random.random(d2.shape))
            #d3 = Sigmoid.f(numx.dot(d2-o2,W2) + b3)
            #d3 = numx.float64(d3 > numx.random.random(d3.shape))
            
        #negative phase
        for _ in range(k_neg):  
            m2 = binary_unit(numx.dot(m1-o1,W1) + numx.dot(m3-o3,W2.T) + b2)
            m1 = gaussian_unit(numx.dot(m2-o2,W1.T) + b1)
            m3 = binary_unit(numx.dot(m2-o2,W2) + b3)
            '''
            m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(m3-o3,W2.T) + b2)
            m2 = numx.float64(m2 > numx.random.random(m2.shape))
            m1 = numx.dot(m2-o2,W1.T) + b1
            m1 = m1 + numx.random.randn(m1.shape[0],m1.shape[1])
            m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
            m3 = numx.float64(m3 > numx.random.random(m3.shape))
            '''
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
        
        dW1 *= mom_W1
        dW2 *= mom_W2
       
        db1 *= mom_b1
        db1 *= mom_b2
        db1 *= mom_b3
        
        # Calculate gradients
        dW1 += (numx.dot((d1-o1).T,d2-o2)-numx.dot((m1-o1).T,m2-o2))
        dW2 += (numx.dot((d2-o2).T,d3-o3)-numx.dot((m2-o2).T,m3-o3))
        
        db1 += (numx.sum(d1-m1,axis = 0)).reshape(1,N)
        db2 += (numx.sum(d2-m2,axis = 0)).reshape(1,M)
        db3 += (numx.sum(d3-m3,axis = 0)).reshape(1,O)

        #db2 += 0.01-new_o2
        #db3 += 0.01-new_o3

        #dW1 = numxExt.restrict_norms(dW1, 1.0, axis = 0)
        #dW2 = numxExt.restrict_norms(dW2, 1.0, axis = 0)
        
        # Update Model
        #W1 += lr_W1/batch_size*dW1
        W1 += lr_W1*numxExt.restrict_norms(dW1/batch_size, 0.1, axis = 0)#*conn1
        #W2 += lr_W2*numxExt.restrict_norms(dW2/batch_size, 0.1, axis = 0)*conn2
        #W2 -= lr_W2*0.002*numx.sign(W2)
        #W2 *= conn2
        #W2 += lr_W2/batch_size*dW2
        
        b1 += lr_b1/batch_size*db1
        b2 += lr_b2/batch_size*db2
        b3 += lr_b3/batch_size*db3
        


    # Estimate Meanfield reconstruction 
    e2 = Sigmoid.f(numx.dot(data-o1,W1) + b2)
    e1 = numx.dot(e2-o2,W1.T) + b1
    e3 = Sigmoid.f(numx.dot(e2-o2,W2) + b3)
    e2 = Sigmoid.f(numx.dot(e1-o1,W1) + numx.dot(e3-o3,W2.T) + b2)
    e1 = numx.dot(e2-o2,W1.T) + b1
    
    # Plot Error and parameter norms, should be the same for both variants
    print numx.mean((e1-data)**2),
    print numx.mean(numxExt.get_norms(W1)),'\t',numx.mean(numxExt.get_norms(W2)),'\t',
    print numx.mean(numxExt.get_norms(b1+numx.dot(0.0-o2,W1.T))),'\t',
    print numx.mean(numxExt.get_norms(b2+numx.dot(0.0-o1,W1) + numx.dot(0.0-o3,W2.T))),'\t',
    print numx.mean(numxExt.get_norms(b3+numx.dot(0.0-o2,W2))),'\t',numx.mean(o1),'\t',numx.mean(new_o2),'\t',numx.mean(new_o3)

# End time measure
measurer.end()
print
print 'End-time: \t', measurer.get_end_time()
print 'Training time:\t', measurer.get_interval()

# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(W1, v11_org,v12_org, v21,v22, border_size = 1,normalized = True), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(W1,W2), v11_org,v12_org, v31,v32, border_size = 1,normalized = True), 'Weights 2')

# Show some sampling
VIS.imshow_matrix(VIS.tile_matrix_columns(pca.unproject(d1), v11_org,v12_org, batch_size1, batch_size2, 1, False),'d')
m2 = Sigmoid.f(numx.dot(d1-o1,W1) + numx.dot(d3-o3,W2.T) + b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = numx.dot(m2-o2,W1.T) + b1
VIS.imshow_matrix(VIS.tile_matrix_columns(pca.unproject(m1), v11_org,v12_org, batch_size1, batch_size2, 1, False),'m 1')

m1 = m1 + numx.random.randn(m1.shape[0],m1.shape[1])
m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
m3 = numx.float64(m3 > numx.random.random(m3.shape))
m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(d3-o3,W2.T) + b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = numx.dot(m2-o2,W1.T) + b1
VIS.imshow_matrix(VIS.tile_matrix_columns(pca.unproject(m1), v11_org,v12_org, batch_size1, batch_size2, 1, False),'m 2')

m1 = m1 + numx.random.randn(m1.shape[0],m1.shape[1])
m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
m3 = numx.float64(m3 > numx.random.random(m3.shape))
m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(d3-o3,W2.T) + b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = numx.dot(m2-o2,W1.T) + b1
VIS.imshow_matrix(VIS.tile_matrix_columns(pca.unproject(m1), v11_org,v12_org, batch_size1, batch_size2, 1, False),'m 3')

m1 = m1 + numx.random.randn(m1.shape[0],m1.shape[1])
m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
m3 = numx.float64(m3 > numx.random.random(m3.shape))
m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(d3-o3,W2.T) + b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = numx.dot(m2-o2,W1.T) + b1
VIS.imshow_matrix(VIS.tile_matrix_columns(pca.unproject(m1), v11_org,v12_org, batch_size1, batch_size2, 1, False),'m 4')


VIS.show()