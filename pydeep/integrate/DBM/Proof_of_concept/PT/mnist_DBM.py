import numpy as numx
import pydeep.base.numpyextension as npExt
from DBM3LayerBinary import Estimator as ESTIMATOR
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.measuring as MEASURE
import pydeep.misc.io as IO
import pydeep.misc.visualization as VIS
import pydeep.misc.toyproblems as PROBLEMS
import mkl

mkl.set_num_threads(4)
# Set the same seed value for all algorithms
numx.random.seed(42)

# Load Data
train_set,train_lab,valid_set,valid_lab,test_set,test_lab = IO.load_MNIST("../../../../../workspacePy/data/mnist.pkl.gz",True)

data = numx.vstack((train_set,valid_set))

v11 = v12 = 28
v21 = 25
v22 = 20
v31 = 25
v32 = 10

N = v11 * v12
M = v21 * v22
O = v31 * v32

offset_typ = '000'

dbm = IO.load_object( offset_typ+"_"+str(M)+"_"+str(O)+".dbm")

a = numx.linspace(0.0, 0.5, 500+1)
a = a[0:a.shape[0]-1]
b = numx.linspace(0.5, 0.9, 4000+1)
b = b[0:b.shape[0]-1]
c = numx.linspace(0.9, 1.0, 10000)
betas = numx.hstack((a,b,c))
#lnZ_AIS, lnZplus, lnZminus = ESTIMATOR.partition_function_AIS(dbm,betas=betas)
#print lnZ_AIS, lnZplus, lnZminus
if offset_typ is 'DDD':
    lnZ_AIS = 1098.5149486

if offset_typ is '000':
    lnZ_AIS = 453.278824813

print "LOWER+AIS  : ",numx.mean(ESTIMATOR._LL_lower_bound_check(dbm,test_set,lnZ_AIS))
print "LOWER+AIS  : ",numx.mean(ESTIMATOR.LL_lower_bound(dbm,data,lnZ_AIS))
exit()
