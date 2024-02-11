import numpy as numx
import model as MODEL
import pydeep.rbm.trainer as TRAINER
import pydeep.rbm.estimator as ESTIMATOR

import pydeep.misc.visualization as STATISTICS
import pydeep.misc.io as IO
import pydeep.misc.visualization as VISUALIZATION
import pydeep.misc.preprocessing as PREPROCESSING
import pydeep.misc.numpyextension as npExt
import pydeep.misc.measuring as MEASURE

import mkl
import scipy.io as sio
mkl.set_num_threads(2)

from CheckedAIS import annealed_importance_sampling as AISCHECK
from oldAIS import annealed_importance_sampling as AISOLD

## evaluate model
a = numx.linspace(0.0, 0.5, 100+1)
a = a[0:a.shape[0]-1]
b = numx.linspace(0.5, 0.9, 500+1)
b = b[0:b.shape[0]-1]
c = numx.linspace(0.9, 1.0, 1000+1)
betas = numx.hstack((a,b,c))


# Model Parameters
h1 = 4
h2 = 4
v1 = 14
v2 = 14

# Load and whiten data
data = numx.random.permutation(IO.load_matlab_file('../../../workspacePy/data/NaturalImage.mat','rawImages'))
data = PREPROCESSING.remove_rows_means(data)
zca = PREPROCESSING.ZCA(v1*v2)
zca.train(data)
data = zca.project(data)

# Training paramters
batch_size = 100
epochs = 21
k = 1
eps = [0.1,0.0,0.1,0.01]
mom = 0.9
decay = 0.0
max_norm = 0.01*numx.max(npExt.get_norms(data, axis = 1))

# Create trainer and model
rbm = MODEL.GaussianBinaryVarianceRBM(number_visibles = v1*v2, 
                                      number_hiddens = h1*h2, 
                                      data=None, 
                                      initial_weights=0.01, 
                                      initial_visible_bias=0.0, 
                                      initial_hidden_bias=-4.0, 
                                      initial_visible_offsets=0.0, 
                                      initial_hidden_offsets=0.0)

# Create trainer
trainer = TRAINER.CD(rbm)
measurer = MEASURE.Stopwatch()

# Train model
print 'Training'
print 'Epoch\tRecon. Error\tLog likelihood \tExpected End-Time'
batch_size = 100
for epoch in range(0,epochs) :
    for b in range(0,data.shape[0],batch_size):
        batch = data[b:b+batch_size,:]    
        trainer.train(data = batch,
                      num_epochs=1, 
                      epsilon=eps, 
                      k=k, 
                      momentum=mom,  
                      desired_sparseness=None, 
                      update_visible_offsets=0.0, 
                      update_hidden_offsets=0.0, 
                      restrict_gradient=max_norm, 
                      restriction_norm='Cols', 
                      use_hidden_states=False,
                      use_centered_gradient = False)
    
    print epoch

    # Calculate Log-Likelihood every 10th epoch
    if(epoch % 10 == 0):
        Z = ESTIMATOR.partition_function_factorize_h(rbm, batchsize_exponent=h1, status = False)
        LL = numx.mean(ESTIMATOR.log_likelihood_v(rbm,Z, data))
        RE = numx.mean(ESTIMATOR.reconstruction_error(rbm, data)) 
        print '%d\t%8.6f\t%8.4f\t' % (epoch, RE, LL),
        print measurer.get_expected_end_time(epoch+1, epochs),
        print

measurer.end()

# Plot Likelihood and partition function calculate with different methods
print
print 'End-time: \t', measurer.get_end_time()
print 'Training time:\t', measurer.get_interval()

#Z_via_h = ESTIMATOR.partition_function_factorize_h(rbm, batchsize_exponent=h1, status= False)
Z_ais, Z_min, Z_max = AISOLD(rbm, num_runs=100, betas = betas)
print
print 'Z ais: %4.8f' % Z_ais
print
print 'Z true: %4.8f' % ESTIMATOR.partition_function_factorize_h(rbm, batchsize_exponent=h1, status = False)

# Prepare results
rbmReordered = STATISTICS.reorder_filter_by_hidden_activation(rbm, data)
VISUALIZATION.imshow_matrix(VISUALIZATION.tile_matrix_rows(zca.unproject(rbmReordered.w.T).T, v1,v2, h1, h2, border_size = 1,normalized = True), 'Weights')
samples = STATISTICS.generate_samples(rbmReordered, data[0:30], 30, 1, v1, v2, False, zca)
VISUALIZATION.imshow_matrix(samples,'Samples')

# Display results
subsetFilters = rbmReordered.w[:,0:20]
opt_frq, opt_ang = STATISTICS.filter_frequency_and_angle(subsetFilters)
VISUALIZATION.imshow_filter_tuning_curve(subsetFilters)
VISUALIZATION.imshow_filter_frequency_angle_histogram(opt_frq, opt_ang)
VISUALIZATION.imshow_filter_optimal_gratings(subsetFilters, opt_frq, opt_ang)

VISUALIZATION.show()




