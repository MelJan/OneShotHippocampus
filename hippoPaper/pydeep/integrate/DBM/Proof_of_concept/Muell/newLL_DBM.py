import numpy as numx
import pydeep.base.numpyextension as npExt
from DBM3LayerBinary import Model as MODEL
from DBM3LayerBinary import Trainer as TRAINER
from DBM3LayerBinary import Estimator as ESTIMATOR
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.measuring as MEASURE
import pydeep.misc.io as IO
import pydeep.misc.visualization as VIS
import pydeep.misc.toyproblems as PROBLEMS

# Set the same seed value for all algorithms
numx.random.seed(42)

# Load Data
data = PROBLEMS.generate_bars_and_stripes_complete(3)
data = numx.vstack((data[0],data,data[data.shape[0]-1]))
'''
#data = numx.array([[0,0,0],[1,1,0],[0,1,1],[1,0,1]])
0.0
EXACT 1    :  11.783735541
EXACT 2    :  11.783735541
AIS        :  11.7837351368
EXACT+EXACT:  -99.8132189892
EXACT+EXACT:  -99.8132189892
LOWER+EXACT:  -99.8152132224
LOWER+EXACT:  -99.8152132224
LOWER+AIS  :  -99.8152067543
0.01
EXACT 1    :  11.7881503257
EXACT 2    :  11.7881503257
AIS        :  11.7881663604
EXACT+EXACT:  -99.6833466457
EXACT+EXACT:  -99.6833466457
LOWER+EXACT:  -99.7328833734
LOWER+EXACT:  -99.7328833734
LOWER+AIS  :  -99.7331399279
0.02
EXACT 1    :  12.1122094157
EXACT 2    :  12.1122094157
AIS        :  12.1117870923
EXACT+EXACT:  -93.6589316946
EXACT+EXACT:  -93.6589316946
LOWER+EXACT:  -93.7410192935
LOWER+EXACT:  -93.7410192935
LOWER+AIS  :  -93.7342621187
0.03
EXACT 1    :  12.6379961534
EXACT 2    :  12.6379961534
AIS        :  12.6405330115
EXACT+EXACT:  -91.1108892024
EXACT+EXACT:  -91.1108892024
LOWER+EXACT:  -91.1656107626
LOWER+EXACT:  -91.1656107626
LOWER+AIS  :  -91.2062004921
0.04
'''
# Set dimensions
v11 = 3
v12 = 3
v21 = 2
v22 = 2
v31 = 2
v32 = 2

N = v11 * v12
M = v21 * v22
O = v31 * v32

# Training parameters
batch_size = data.shape[0]
epochs = 100000
epsilon = 2*numx.array([0.01,0.1,0.0,0.01,0.0,0.0,0.01,0.01])
k = [10,10]
offset_typ = 'DDD'
meanfield=True
                  
dbm = MODEL(N,M,O,offset_typ,data)
#dbm.o1 *= 0.0
#dbm.o2 *= 0.0
#dbm.b3 += -999
trainer = TRAINER(dbm,batch_size)

# Start time measure and training
measurer = MEASURE.Stopwatch()
for epoch in range(0,epochs) :
    
    for b in range(0,data.shape[0],batch_size):
        trainer.train(data = data[b:b+batch_size,:],
                  epsilon = epsilon,
                  k = k,
                  offset_typ = offset_typ, 
                  meanfield=meanfield)
    # Estimate Meanfield reconstruction 
    e2 = Sigmoid.f(numx.dot(data-dbm.o1,dbm.W1) + numx.dot(-dbm.o3,dbm.W2.T) + dbm.b2)
    e1 = Sigmoid.f(numx.dot(e2-dbm.o2,dbm.W1.T) + dbm.b1)
    e3 = Sigmoid.f(numx.dot(e2-dbm.o2,dbm.W2) + dbm.b3)
    e2 = Sigmoid.f(numx.dot(e1-dbm.o1,dbm.W1) + numx.dot(e3-dbm.o3,dbm.W2.T) + dbm.b2)
    e1 = Sigmoid.f(numx.dot(e2-dbm.o2,dbm.W1.T) + dbm.b1)

    if epoch % 1000 == 0:
        print epoch/numx.float64(epochs)
        lnZ_exact1 = ESTIMATOR.partition_function_exact(dbm)
        lnZ_exact2 = ESTIMATOR._partition_function_exact_check(dbm)
        lnZ_AIS = ESTIMATOR.partition_function_AIS(dbm)[0]
        print "EXACT 1    : ",lnZ_exact1
        print "EXACT 2    : ",lnZ_exact2
        print "AIS        : ",lnZ_AIS
        print "EXACT+EXACT: ",numx.sum(ESTIMATOR.LL_exact(dbm,data,lnZ_exact1))
        print "EXACT+EXACT: ",numx.sum(ESTIMATOR._LL_exact_check(dbm,data,lnZ_exact1))
        print "LOWER+EXACT: ",numx.sum(ESTIMATOR.LL_lower_bound(dbm,data,lnZ_exact1))
        print "LOWER+EXACT: ",numx.sum(ESTIMATOR.LL_lower_bound(dbm,data,lnZ_exact1))
        print "LOWER+AIS  : ",numx.sum(ESTIMATOR.LL_lower_bound(dbm,data,lnZ_AIS))

# End time measure
measurer.end()
print
print 'End-time: \t', measurer.get_end_time()
print 'Training time:\t', measurer.get_interval()


# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(dbm.W1, v11,v12, v21,v22, border_size = 0,normalized = False), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(dbm.W1,dbm.W2), v11,v12, v31,v32, border_size = 0,normalized = False), 'Weights 2')

# Show some sampling
d3 = dbm.o3
VIS.imshow_matrix(VIS.tile_matrix_columns(data, v11, v12, 1, batch_size, 0, False),'d')
m2 = Sigmoid.f(numx.dot(data-dbm.o1,dbm.W1) + numx.dot(d3-dbm.o3,dbm.W2.T) + dbm.b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W1.T) + dbm.b1)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v11, v12, 1, batch_size, 1, False),'m 1')

m1 = numx.float64(m1 > numx.random.random(m1.shape))
m3 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W2) + dbm.b3)
m3 = numx.float64(m3 > numx.random.random(m3.shape))
m2 = Sigmoid.f(numx.dot(m1-dbm.o1,dbm.W1) + numx.dot(d3-dbm.o3,dbm.W2.T) + dbm.b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W1.T) + dbm.b1)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v11, v12, 1, batch_size, 0, False),'m 2')

m1 = numx.float64(m1 > numx.random.random(m1.shape))
m3 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W2) + dbm.b3)
m3 = numx.float64(m3 > numx.random.random(m3.shape))
m2 = Sigmoid.f(numx.dot(m1-dbm.o1,dbm.W1) + numx.dot(d3-dbm.o3,dbm.W2.T) + dbm.b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W1.T) + dbm.b1)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v11, v12, 1, batch_size, 0, False),'m 3')

m1 = numx.float64(m1 > numx.random.random(m1.shape))
m3 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W2) + dbm.b3)
m3 = numx.float64(m3 > numx.random.random(m3.shape))
m2 = Sigmoid.f(numx.dot(m1-dbm.o1,dbm.W1) + numx.dot(d3-dbm.o3,dbm.W2.T) + dbm.b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W1.T) + dbm.b1)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v11, v12, 1, batch_size, 0, False),'m 4')

VIS.show()