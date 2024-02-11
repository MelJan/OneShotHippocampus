import numpy as numx
import pydeep.base.numpyextension as npExt
import DBM3LayerBinary.model as MODEL
import DBM3LayerBinary.trainer as TRAINER
import DBM3LayerBinary.estimator as ESTIMATOR
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.measuring as MEASURE
import pydeep.misc.io as IO
import pydeep.misc.visualization as VIS
import pydeep.misc.toyproblems as PROBLEMS
import mkl

mkl.set_num_threads(8)
# Set the same seed value for all algorithms
numx.random.seed(42)

# Load Data
train_set,train_lab,valid_set,valid_lab,test_set,test_lab = IO.load_MNIST("../../../../workspacePy/data/mnist.pkl.gz",True)

data = numx.vstack((train_set,valid_set))

v11 = v12 = 28
v21 = 25
v22 = 20
v31 = 25
v32 = 20

N = v11 * v12
M = v21 * v22
O = v31 * v32

# Training parameters
folder = "mf/"
batch_size = 100
epochs = 500
method = 'PCD'
k = [1,1]
meanfield=0.0001
offset_typ = 'DDD'

epsilon = 0.001*numx.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
if offset_typ[0] is 'D' or  offset_typ[0] is '0':
    epsilon[5] *= 0.0
if offset_typ[1] is '0':
    epsilon[6] *= 0.0
if offset_typ[2] is '0':
    epsilon[7] *= 0.0
           
dbm = MODEL.BinaryBinaryDBM(N,M,O,offset_typ,data)
if method is 'PCD':
    trainer = TRAINER.PCD(dbm,batch_size)
if method is 'CD':
    trainer = TRAINER.CD(dbm,batch_size)
if method is 'PT-20':
    trainer = TRAINER.PT(dbm,batch_size,20)



dbm = IO.load_object(folder+offset_typ+"_"+str(N)+"x"+str(M)+"x"+str(O)+"_"+str(method)+"_"+str(k[1])+"_"+str(149)+'_'+str(batch_size)+".dbm")


a = numx.linspace(0.0, 0.5, 500+1)
a = a[0:a.shape[0]-1]
b = numx.linspace(0.5, 0.9, 4000+1)
b = b[0:b.shape[0]-1]
c = numx.linspace(0.9, 1.0, 10000)
betas = numx.hstack((a,b,c))
lnZ_AIS =  ESTIMATOR.partition_function_AIS(dbm,betas=betas)


#if offset_typ is '000':
#    lnZ_AIS = 1046.78886065 # 000
#if offset_typ is 'DDD':
#    lnZ_AIS = 1235.66124768 # DDD
    
print lnZ_AIS
#measurer = MEASURE.Stopwatch()
print numx.mean(ESTIMATOR.LL_lower_bound(dbm,test_set,lnZ_AIS))
#print 'Training time:\t', measurer.get_interval()
#measurer = MEASURE.Stopwatch()
#print numx.mean(ESTIMATOR._LL_lower_bound_check(dbm,test_set,lnZ_AIS))
#print 'Training time:\t', measurer.get_interval()
exit()


# Start time measure and training
measurer = MEASURE.Stopwatch()
for epoch in range(0,epochs) :
    for b in range(0,data.shape[0],batch_size):
        trainer.train(data = data[b:b+batch_size,:],
                      epsilon = epsilon,
                      k = k,
                      offset_typ = offset_typ, 
                      meanfield=meanfield)
    print epoch," of ",epochs

    #if epoch % 10 == 99:
    if epoch == 147 or epoch == 148:
        print epoch," of ",epochs
        #dbm = IO.load_object(offset_typ+".dbm")

        a = numx.linspace(0.0, 0.5, 500+1)
        a = a[0:a.shape[0]-1]
        b = numx.linspace(0.5, 0.9, 4000+1)
        b = b[0:b.shape[0]-1]
        c = numx.linspace(0.9, 1.0, 10000)
        betas = numx.hstack((a,b,c))
        lnZ_AIS = ESTIMATOR.partition_function_AIS(dbm,betas=betas)[0]
        print "LOWER+AIS  : ",numx.mean(ESTIMATOR.LL_lower_bound(dbm,data,lnZ_AIS))
        IO.save_object(dbm, folder+offset_typ+"_"+str(N)+"x"+str(M)+"x"+str(O)+"_"+str(method)+"_"+str(k[1])+"_"+str(epoch)+'_'+str(batch_size)+".dbm")


# End time measure
measurer.end()
print
print 'End-time: \t', measurer.get_end_time()
print 'Training time:\t', measurer.get_interval()


# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(dbm.W1, v11,v12, v21,v22, border_size = 1,normalized = False), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(dbm.W1,dbm.W2), v11,v12, v31,v32, border_size = 1,normalized = False), 'Weights 2')

# Show some sampling
d1 = data[0:20,:]
d3 = dbm.o3
VIS.imshow_matrix(VIS.tile_matrix_columns(d1, v11, v12, 1, 20, 1, False),'d')
m2 = Sigmoid.f(numx.dot(d1-dbm.o1,dbm.W1) + numx.dot(d3-dbm.o3,dbm.W2.T) + dbm.b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W1.T) + dbm.b1)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v11, v12, 1, 20, 1, False),'m 1')

m1 = numx.float64(m1 > numx.random.random(m1.shape))
m3 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W2) + dbm.b3)
m3 = numx.float64(m3 > numx.random.random(m3.shape))
m2 = Sigmoid.f(numx.dot(m1-dbm.o1,dbm.W1) + numx.dot(d3-dbm.o3,dbm.W2.T) + dbm.b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W1.T) + dbm.b1)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v11, v12, 1, 20, 1, False),'m 2')

m1 = numx.float64(m1 > numx.random.random(m1.shape))
m3 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W2) + dbm.b3)
m3 = numx.float64(m3 > numx.random.random(m3.shape))
m2 = Sigmoid.f(numx.dot(m1-dbm.o1,dbm.W1) + numx.dot(d3-dbm.o3,dbm.W2.T) + dbm.b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W1.T) + dbm.b1)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v11, v12, 1, 20, 1, False),'m 3')

m1 = numx.float64(m1 > numx.random.random(m1.shape))
m3 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W2) + dbm.b3)
m3 = numx.float64(m3 > numx.random.random(m3.shape))
m2 = Sigmoid.f(numx.dot(m1-dbm.o1,dbm.W1) + numx.dot(d3-dbm.o3,dbm.W2.T) + dbm.b2)
m2 = numx.float64(m2 > numx.random.random(m2.shape))
m1 = Sigmoid.f(numx.dot(m2-dbm.o2,dbm.W1.T) + dbm.b1)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v11, v12, 1, 20, 1, False),'m 4')


IO.save_object(dbm, folder+offset_typ+"_"+str(N)+"x"+str(M)+"x"+str(O)+"_"+str(method)+"_"+str(k[1])+"_"+str(epochs)+'_'+str(batch_size)+".dbm")

a = numx.linspace(0.0, 0.5, 500+1)
a = a[0:a.shape[0]-1]
b = numx.linspace(0.5, 0.9, 4000+1)
b = b[0:b.shape[0]-1]
c = numx.linspace(0.9, 1.0, 10000)
betas = numx.hstack((a,b,c))
lnZ_AIS = ESTIMATOR.partition_function_AIS(dbm,betas=betas)[0]
print "LOWER+AIS  : ",numx.mean(ESTIMATOR.LL_lower_bound(dbm,test_set,lnZ_AIS))
#exit()

VIS.show()