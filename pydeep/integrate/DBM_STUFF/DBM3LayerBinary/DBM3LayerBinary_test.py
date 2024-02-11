import numpy as numx
import pydeep.base.numpyextension as numxExt
import DBM3LayerBinary.model as MODEL
import DBM3LayerBinary.trainer as TRAINER
import DBM3LayerBinary.estimator as ESTIMATOR
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.io as IO
import pydeep.misc.toyproblems as TOY
import pydeep.misc.visualization as VIS
import mkl

mkl.set_num_threads(8)
# Set the same seed value for all algorithms
numx.random.seed(42)

# Load Data
#train_set, _, valid_set, _, test_set, _ = IO.load_mnist("../../../../data/mnist.pkl.gz",True)
#train_set = numx.vstack((train_set,valid_set))


# Set dimensions
v11 = v12 = 2
v21 = v22 = 4
v31 = v32 = 2

a = numx.linspace(0.0, 0.5, 100+1)
a = a[0:a.shape[0]-1]
b = numx.linspace(0.5, 0.9, 800+1)
b = b[0:b.shape[0]-1]
c = numx.linspace(0.9, 1.0, 2000)
betas = numx.hstack((a,b,c))

train_set = test_set = TOY.generate_bars_and_stripes_complete(v11)

N = v11 * v12
M = v21 * v22
O = v31 * v32

# Training parameters
batch_size = train_set.shape[0]
epochs = 100000
k_pos = 3
k_neg = 10
epsilon = 0.01*numx.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
offset_typ = 'DDD'

dbm = MODEL.BinaryBinaryDBM(N,M,O,offset_typ,train_set)

# Set the same seed value for all algorithms
numx.random.seed(42)

# Initialize parameters
dbm.W1 = numx.random.randn(N, M) * 0.01
dbm.W2 = numx.random.randn(M, O) * 0.01

dbm.o1 = numx.mean(train_set, axis = 0).reshape(1,N)
dbm.o2 = numx.zeros((1,M)) + 0.5
dbm.o3 = numx.zeros((1,O)) + 0.5

dbm.b1 = Sigmoid.g(numx.clip(dbm.o1,0.001,0.999))
dbm.b2 = Sigmoid.g(numx.clip(dbm.o2,0.001,0.999))
dbm.b3 = Sigmoid.g(numx.clip(dbm.o3,0.001,0.999))

# Initialize negative Markov chain
dbm.m1 = dbm.o1+numx.zeros((batch_size,N))
dbm.m2 = dbm.o2+numx.zeros((batch_size,M))
dbm.m3 = dbm.o3+numx.zeros((batch_size,O))

trainer = TRAINER.PCD(dbm,batch_size)
numx.random.seed(42)
# Start time measure and training
for epoch in xrange(0,epochs+1) :
    for b in xrange(0,train_set.shape[0],batch_size):
        trainer.train(data = train_set[b:b+batch_size,:],
                      epsilon = epsilon,
                      k = [k_pos,k_neg],
                      offset_typ = offset_typ,
                      meanfield=True)
    if epoch % 1000 == 0:
        print numx.mean(numxExt.get_norms(dbm.W1)),'\t',numx.mean(numxExt.get_norms(dbm.W2)),'\t',
        print numx.mean(numxExt.get_norms(dbm.b1-numx.dot(dbm.o2,dbm.W1.T))),'\t',numx.mean(numxExt.get_norms(dbm.b2-numx.dot(dbm.o1,dbm.W1)-numx.dot(dbm.o3,dbm.W2.T))),'\t',
        print numx.mean(numxExt.get_norms(dbm.b3-numx.dot(dbm.o2,dbm.W2))),'\t',numx.mean(dbm.o1),'\t',numx.mean(dbm.o2),'\t',numx.mean(dbm.o3)

        logZ, logZ_up, logZ_down = ESTIMATOR.partition_function_AIS(trainer.model, betas=betas)
        test_LL = numx.mean(ESTIMATOR.LL_lower_bound(trainer.model, test_set, logZ))
        train_LL = numx.mean(ESTIMATOR.LL_lower_bound(trainer.model, train_set, logZ))
        print(2**(v11+1) * train_LL, 2**(v11+1)* test_LL)

        logZ = ESTIMATOR.partition_function_exact(trainer.model)
        test_LL = numx.mean(ESTIMATOR.LL_exact(trainer.model, test_set, logZ))
        train_LL = numx.mean(ESTIMATOR.LL_exact(trainer.model, train_set, logZ))
        print(2**(v11+1)* train_LL, 2**(v11+1)* test_LL)

# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(dbm.W1, v11,v12, v21,v22, border_size = 1,normalized = False), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(dbm.W1,dbm.W2), v11,v12, v31,v32, border_size = 1,normalized = False), 'Weights 2')






VIS.show()