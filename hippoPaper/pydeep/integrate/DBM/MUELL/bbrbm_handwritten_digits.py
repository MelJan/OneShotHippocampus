''' Example using GB-RBMs on the MNIST handwritten digit database.

    :Version:
        3.0
        
    :Date
        15.07.2014
    
    :Author:
        Jan Melchior
        
    :Contact:
        pydeep@gmail.com
        
    :License:
        
        Copyright (C) 2014  Jan Melchior

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>. 
           
'''
import numpy as numx
import pydeep.rbm.model as MODEL
import pydeep.rbm.trainer as TRAINER
import pydeep.rbm.estimator as ESTIMATOR

import pydeep.misc.visualization as STATISTICS
import pydeep.misc.io as IO
import pydeep.misc.visualization as VISUALIZATION
import pydeep.misc.measuring as MEASURE

# Set random seed (optional)
numx.random.seed(42)

# Model Parameters
h1 = 16
h2 = 16
v1 = 28
v2 = 32
v3 = 16*7
# Load data and whiten it
train_set = numx.hstack((IO.load_MNIST("../../../workspacePy/data/mnist.pkl.gz",False)[0],numx.ones((50000,v3))*0.5))

# Training paramters
batch_size = 100
epochs = 5
k = 1
eps = 0.05
mom = 0.0
decay = 0.0
update_visible_mean = 0
update_hidden_mean = 0

# Create trainer and model
rbm = MODEL.BinaryBinaryRBM(number_visibles = v1*v2, 
                            number_hiddens = h1*h2, 
                            data=train_set, 
                            initial_weights=0.01, 
                            initial_visible_bias=0, 
                            initial_hidden_bias=0, 
                            initial_visible_offsets=0, 
                            initial_hidden_offsets=0)
trainer = TRAINER.PCD(rbm,25)
measurer = MEASURE.Stopwatch()

# Train model
print 'Training'
print 'Epoch\tRecon. Error\tLog likelihood \tExpected End-Time'
for epoch in range(0,epochs) :
    for b in range(0,train_set.shape[0],batch_size):
        batch = numx.hstack((train_set[b:b+batch_size,0:rbm.input_dim-v3],numx.ones((batch_size,v3))*rbm.ov[:,rbm.input_dim-v3:rbm.input_dim]))
        batch = rbm.probability_v_given_h(rbm.probability_h_given_v(batch))
        batch = numx.hstack((train_set[b:b+batch_size,0:rbm.input_dim-v3],numx.ones((batch_size,v3))*batch[:,rbm.input_dim-v3:rbm.input_dim]))
        trainer.train(data = batch,
                      num_epochs=1, 
                      epsilon=eps, 
                      k=k, 
                      momentum=mom, 
                      update_visible_offsets=update_visible_mean, 
                      update_hidden_offsets=update_hidden_mean)

    print epoch
    if(epoch == 5):
        mom = 0.0
        

    RE = numx.mean(ESTIMATOR.reconstruction_error(rbm, train_set)) 
    print '%d\t%8.6f' % (epoch, RE),
    print measurer.get_expected_end_time(epoch+1, epochs),
    print

measurer.end()

# Plot Likelihood and partition function calculate with different methods
print
print 'End-time: \t', measurer.get_end_time()
print 'Training time:\t', measurer.get_interval()

#Z_ais = ESTIMATOR.annealed_importance_sampling(rbm, status= False)[0]
#print
#print "AIS  Partition: ", Z_ais," ( LL: ",
#print numx.mean(ESTIMATOR.log_likelihood_v(rbm, Z_ais , train_set))

# Prepare results
rbmReordered = STATISTICS.reorder_filter_by_hidden_activation(rbm, train_set)
VISUALIZATION.imshow_matrix(VISUALIZATION.tile_matrix_rows(rbm.w[0:784,:], 28,28, h1, h2, border_size = 1,normalized = True), 'Weights')

VISUALIZATION.imshow_matrix(VISUALIZATION.tile_matrix_rows(numx.dot(rbm.w[0:784,:],rbm.w[784:896,:].T), 28,28, 16, 7, border_size = 1,normalized = True), 'Weights 2')
VISUALIZATION.imshow_standard_rbm_parameters(rbmReordered, v1,v2,h1, h2)
samples = STATISTICS.generate_samples(rbm, train_set[0:30], 30, 1, v1, v2, False, None)
VISUALIZATION.imshow_matrix(samples,'Samples')

VISUALIZATION.show()
