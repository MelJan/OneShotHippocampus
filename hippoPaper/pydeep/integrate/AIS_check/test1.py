import numpy as numx
import pydeep.rbm.model as MODEL
import pydeep.rbm.trainer as TRAINER
import pydeep.rbm.estimator as ESTIMATOR
import pydeep.misc.visualization as VISUALIZATION
import pydeep.misc.io as IO
import pydeep.misc.toyproblems as TOY

import pydeep.misc.numpyextension as npExt
import mkl
import scipy.io as sio
mkl.set_num_threads(2)

from CheckedAIS import annealed_importance_sampling as AISCHECK
from oldAIS import annealed_importance_sampling as AISOLD

#train_data = IO.load_MNIST("../../../workspacePy/data/mnist.pkl.gz",True)[0]

#rbm = IO.load_object(method+'_'+trainMethod+'_'+str(trial)+'_'+str(10)+'_'+str(epsilon)+'_'+str(regL2Norm)+'.rbm')

# Model Parameters
h1 = 4
h2 = 4
v1 = 4
v2 = 4
## evaluate model
a = numx.linspace(0.0, 0.5, 1000+1)
a = a[0:a.shape[0]-1]
b = numx.linspace(0.5, 0.9, 5000+1)
b = b[0:b.shape[0]-1]
c = numx.linspace(0.9, 1.0, 10000+1)
betas = numx.hstack((a,b,c))
# Load data and whiten it
train_set = TOY.generate_bars_and_stripes_complete(v1)#IO.load_MNIST("../../../workspacePy/data/mnist.pkl.gz",False)[0]
print train_set.shape
# Training paramters
batch_size = 1
epochs = 5001
numx.random.seed(42)
# Create trainer and model
rbm = MODEL.BinaryBinaryRBM(number_visibles = v1*v2, 
                            number_hiddens = h1*h2, 
                            data=train_set, 
                            initial_weights=0.01, 
                            initial_visible_bias=0.0, 
                            initial_hidden_bias=0.0, 
                            initial_visible_offsets=0.0, 
                            initial_hidden_offsets=0.0)
trainer = TRAINER.CD(rbm)
numx.random.seed(42)
# Train model
print 'Training'
print 'Epoch\tRecon. Error\tLog likelihood \tExpected End-Time'
for epoch in range(0,epochs) :
    for b in range(0,train_set.shape[0],batch_size):
        batch = train_set[b:b+batch_size,:]
        trainer.train(data = batch,
                          num_epochs=1, 
                          epsilon=0.1, 
                          k=1, 
                          momentum=0.0, 
                          regL1Norm=0.0, 
                          regL2Norm=0.0, 
                          regSparseness=0.0, 
                          desired_sparseness=None, 
                          update_visible_offsets=0.0, 
                          update_hidden_offsets=0.0, 
                          offset_typ='DD', 
                          use_centered_gradient=False, 
                          restrict_gradient=False, 
                          restriction_norm='Mat', 
                          use_hidden_states=False)

    
    if epoch % 500 == 0:
        #print epoch, numx.mean(ESTIMATOR.reconstruction_error(rbm, train_set))
        numx.random.seed(42)
        #logZ = AISOLD(rbm, num_runs=100, betas = betas, k= 1)[0][0]
        #print logZ , '\t', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ, train_set))   
        numx.random.seed(42)
        logZ = AISCHECK(rbm, num_runs=100, betas = betas, data = train_set)[0][0]
        print logZ , '\t', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ, train_set))  
        numx.random.seed(42)
        logZ = AISCHECK(rbm, num_runs=100, betas = betas, data = None)[0][0]
        print logZ , '\t', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ, train_set))   
        numx.random.seed(42)
        logZ = ESTIMATOR.annealed_importance_sampling(rbm, 100, 1, betas, False)[0][0]
        print logZ , '\t', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ, train_set))   
        logZ = ESTIMATOR.partition_function_factorize_h(rbm)
        print logZ , '\t', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ, train_set))         
        



numx.random.seed(42)
logZ = AISCHECK(rbm, num_runs=100, betas = betas,data = train_data)[0]
print logZ

numx.random.seed(42)
logZ = AISOLD(rbm, num_runs=100, betas = betas,data = train_data)[0]
print logZ       

numx.random.seed(42)
logZ = ESTIMATOR.partition_function_factorize_h(rbm)
print logZ                  
