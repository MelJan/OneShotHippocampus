import numpy as numx
import pydeep.base.numpyextension as npExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.rbm.model as MODEL
import pydeep.rbm.estimator as ESTIMATOR
import pydeep.misc.measuring as MEASURE
#import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO
import pydeep.misc.toyproblems as PROBLEMS
    

def energy(x,h1,h2,W1,W2,b1,b2,b3,o1,o2,o3):
    xtemp = x-o1
    h1temp = h1-o2
    h2temp = h2-o3
    return (- numx.dot(xtemp, b1.T)
            - numx.dot(h1temp, b2.T) 
            - numx.dot(h2temp, b3.T) 
            - numx.sum(numx.dot(xtemp, W1) * h1temp,axis=1).reshape(h1temp.shape[0], 1)
            - numx.sum(numx.dot(h1temp, W2) * h2temp,axis=1).reshape(h2temp.shape[0], 1))
    
def naive_LL(x,W1,W2,b1,b2,b3,o1,o2,o3,lnZ):
    all_h1 = npExt.generate_binary_code(W2.shape[0])
    all_h2 = npExt.generate_binary_code(W2.shape[1])
    result = numx.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(all_h1.shape[0]):
            for k in range(all_h2.shape[0]):
                result[i] += numx.exp(-energy(
                                     x[i].reshape(1,x.shape[1]),
                                     all_h1[j].reshape(1,all_h1.shape[1]),
                                     all_h2[k].reshape(1,all_h2.shape[1]),
                                     W1,W2,b1,b2,b3,o1,o2,o3))
    return numx.log(result) - lnZ
    
def unnormalized_log_probability_x(x,h1,W1,W2,b1,b2,b3,o1,o2,o3):
    temp = x - o1
    bias = numx.dot(temp, b1.T).reshape(temp.shape[0], 1)
    

    
    activation = numx.dot(numx.dot(all_h1-o2, W1.T) + b2,temp.T) + numx.dot(numx.dot(all_h2-o3, W2.T) + b3,activation1.T)
    
    
    
    factorx = numx.sum(
                        numx.log(
                                 numx.exp(activation*(1.0 - o3))
                               + numx.exp(-activation*o3)
                                 ) 
                        , axis=1).reshape(temp.shape[0], 1)   
    activation = numx.dot(temp, W2) + b3  
    factorh2 = numx.sum(
                        numx.log(
                                 numx.exp(activation*(1.0 - o3))
                               + numx.exp(-activation*o3)
                                 ) 
                        , axis=1).reshape(temp.shape[0], 1)  
    return bias + factorx + factorh2

def unnormalized_log_probability_h1(h1,W1,W2,b1,b2,b3,o1,o2,o3):
    temp = h1 - o2
    bias = numx.dot(temp, b2.T).reshape(temp.shape[0], 1)
    activation = numx.dot(temp, W1.T) + b1
    factorx = numx.sum(
                        numx.log(
                                 numx.exp(activation*(1.0 - o1))
                               + numx.exp(-activation*o1)
                                 ) 
                        , axis=1).reshape(temp.shape[0], 1)   
    activation = numx.dot(temp, W2) + b3  
    factorh2 = numx.sum(
                        numx.log(
                                 numx.exp(activation*(1.0 - o3))
                               + numx.exp(-activation*o3)
                                 ) 
                        , axis=1).reshape(temp.shape[0], 1)  
    return bias + factorx + factorh2

def partition_function_factorize_h1(W1,W2,b1,b2,b3,o1,o2,o3,batchsize_exponent='AUTO'):

    bit_length = W1.shape[1]
    if batchsize_exponent is 'AUTO' or batchsize_exponent > 20:
        batchsize_exponent = numx.min([W1.shape[1], 12])
    batchSize = numx.power(2, batchsize_exponent)
    num_combinations = numx.power(2, bit_length)
    num_batches = num_combinations / batchSize
    bitCombinations = numx.zeros((batchSize, W1.shape[1]))
    log_prob_vv_all = numx.zeros(num_combinations)

    for batch in range(1, num_batches + 1):
        # Generate current batch
        bitCombinations = npExt.generate_binary_code(bit_length, 
                                                     batchsize_exponent, 
                                                     batch - 1)
        # calculate LL
        log_prob_vv_all[(batch - 1) * batchSize:batch * batchSize] = unnormalized_log_probability_h1(bitCombinations,W1,W2,b1,b2,b3,o1,o2,o3).reshape(
                                                    bitCombinations.shape[0])
    # return the log_sum of values
    return npExt.log_sum_exp(log_prob_vv_all)


# Set the same seed value for all algorithms
numx.random.seed(42)


# Load Data
data = PROBLEMS.generate_bars_and_stripes_complete(3)
data = numx.vstack((data[0],data,data[data.shape[0]-1]))
# Set dimensions
v11 = v12 = 3
v21 = v22 = 2
v31 = v32 = 2

N = v11 * v12
M = v21 * v22
O = v31 * v32

# Training parameters
batch_size = data.shape[0]
epochs = 5000
k_pos = 3
k_neg = 1

lr_W1 = 0.01
lr_W2 = 0.01

lr_b1 = 0.01
lr_b2 = 0.01
lr_b3 = 0.01

lr_o1 = 0.0
lr_o2 = 0.1
lr_o3 = 0.05

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
print numx.sum(numx.abs(W1))
print numx.sum(numx.abs(W2))
print numx.sum(numx.abs(b1))
print numx.sum(numx.abs(b2))
print numx.sum(numx.abs(b3))
print numx.sum(numx.abs(o1))
print numx.sum(numx.abs(o2))
print numx.sum(numx.abs(o3))
for epoch in range(0,epochs) :
    for b in range(0,data.shape[0],batch_size):
        
        #print b/numx.float64(data.shape[0])

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
    e2 = Sigmoid.f(numx.dot(data-o1,W1) + numx.dot(-o3,W2.T) + b2)
    e1 = Sigmoid.f(numx.dot(e2-o2,W1.T) + b1)
    e3 = Sigmoid.f(numx.dot(e2-o2,W2) + b3)
    e2 = Sigmoid.f(numx.dot(e1-o1,W1) + numx.dot(e3-o3,W2.T) + b2)
    e1 = Sigmoid.f(numx.dot(e2-o2,W1.T) + b1)

    # Plot Error and parameter norms, should be the same for both variants
   #print numx.mean((e1-data)**2),
   # print numx.mean(npExt.get_norms(W1)),'\t',numx.mean(npExt.get_norms(W2)),'\t',
   # print numx.mean(npExt.get_norms(b1+numx.dot(0.0-o2,W1.T))),'\t',
   # print numx.mean(npExt.get_norms(b2+numx.dot(0.0-o1,W1) + numx.dot(0.0-o3,W2.T))),'\t',
   # print numx.mean(npExt.get_norms(b3+numx.dot(0.0-o2,W2))),'\t',numx.mean(o1),'\t',numx.mean(o2),'\t',numx.mean(o3)



# End time measure
measurer.end()
print
print 'End-time: \t', measurer.get_end_time()
print 'Training time:\t', measurer.get_interval()

'''
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
'''
# Set the same seed value for all algorithms
numx.random.seed(42)
rbm = MODEL.BinaryBinaryRBM(N+O,M)
rbm.w *= 0
rbm.bv *= 0
rbm.bh *= 0
rbm.ov *= 0
rbm.oh *= 0
rbm.w += numx.vstack((W1,W2.T))
rbm.bv += numx.hstack((b1,b3))
rbm.ov += numx.hstack((o1,o3))
rbm.bh += b2
rbm.oh += o2
# Set the same seed value for all algorithms
numx.random.seed(42)
lnZ_RBM = ESTIMATOR.partition_function_factorize_h(rbm)
# Set the same seed value for all algorithms
numx.random.seed(42)
lnZ_test = partition_function_factorize_h1(W1,W2,b1,b2,b3,o1,o2,o3)
print lnZ_test
print lnZ_RBM
print numx.sum(naive_LL(data,W1,W2,b1,b2,b3,o1,o2,o3,lnZ_RBM))
print numx.sum(naive_LL(data,W1,W2,b1,b2,b3,o1,o2,o3,lnZ_test))

#VIS.show()