''' Toy example using GB-RBMs on a blind source seperation toy problem.

    :Version:
        1.0

    :Date:
        06.06.2016

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2016 Jan Melchior

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

import pydeep.base.numpyextension as npExt
import pydeep.misc.toyproblems as TOY_DATA
import pydeep.misc.visualization as VISUALIZATION
import pydeep.rbm.estimator as ESTIMATOR
import pydeep.rbm.model as MODEL
import pydeep.rbm.trainer as TRAINER
import pydeep.preprocessing as pre

numx.random.seed(42)

# Create mixture 
#data = TOY_DATA.generate_2d_mixtures(50000, -5, 1.0)[0]
data = TOY_DATA.generate_2d_mixtures(50000, 5, 1.0)[0]
#data = numx.vstack((numx.random.multivariate_normal([+1,+1],[[0.1,0],[0,0.1]],5000),
#                    numx.random.multivariate_normal([+1,-1],[[0.1,0],[0,0.1]],5000),
                    #numx.random.multivariate_normal([0,0],[[0.1,0],[0,0.1]],5000),
#                    numx.random.multivariate_normal([-1,-1],[[0.1,0],[0,0.1]],5000),
  #                  numx.random.multivariate_normal([-1,+1],[[0.1,0],[0,0.1]],5000)))
#
data = numx.random.permutation(data)
data = numx.random.permutation(data)

# Whiten data
zca = pre.ZCA(data.shape[1])
zca.train(data)
whiteData = zca.project(data)
#whiteData = data#-numx.array([2.0,1.0])

# split training test data
train_data = whiteData[0:numx.int32(whiteData.shape[0]/2.0),:]
test_data = whiteData[numx.int32(whiteData.shape[0]/2.0):whiteData.shape[0],:]

# Input output dims
h1 = 2
h2 = 2
v1 = whiteData.shape[1]
v2 = 1

# Create model
rbm = MODEL.GaussianBinaryVarianceRBM(number_visibles = v1*v2, 
                                      number_hiddens = h1*h2, 
                                      data=train_data,
                                      initial_weights='AUTO',
                                      initial_visible_bias=0,
                                      initial_hidden_bias=0,
                                      initial_sigma='AUTO',
                                      initial_visible_offsets=0.0,
                                      initial_hidden_offsets=0.0,
                                      dtype=numx.float64)
#rbm.bh = -(npExt.get_norms(rbm.w+rbm.bv.T, axis = 0)-npExt.get_norms(rbm.bv, axis = None))/2.0+numx.log(0.1)
#rbm.bh = rbm.bh.reshape(1,h1*h2)

# Create trainer
trainer = TRAINER.PCD(rbm,100)

# Hyperparameter
batch_size = 100
max_epochs = 100
k = 1
epsilon = [0.1,0.1,0.1,0.01]
momentum = 0.9
weight_decay = 0
update_visible_mean=0.01
update_hidden_mean=0.01
desired_sparseness=0.0
restrict_gradient= False#0.01*numx.max(npExt.get_norms(train_data, axis = 1))
use_hidden_states=False
use_enhanced_gradient=False
biases = []
means = []
weights = []
# Train model
print 'Training'
print 'Epoch\tRE train_images\tRE test \tLL train_images\tLL test '
for epoch in range(1,max_epochs+1) :
    # Shuffle data points
    train_data = numx.random.permutation(train_data)
    # loop over batches
    for b in range(0,train_data.shape[0]/batch_size) :
        trainer.train(data = train_data[b:(b+batch_size),:],
                      num_epochs=1, 
                      epsilon=epsilon, 
                      k=k, 
                      momentum=momentum, 
                      reg_l1norm=0.0,
                      reg_l2norm=0.0,
                      reg_sparseness = 0.0,
                      desired_sparseness=desired_sparseness, 
                      update_visible_offsets=update_visible_mean, 
                      update_hidden_offsets=update_hidden_mean, 
                      restrict_gradient=restrict_gradient, 
                      restriction_norm='Cols',
                      use_hidden_states=use_hidden_states,
                      use_centered_gradient=use_enhanced_gradient)
        biases.append([numx.sqrt(rbm.bv[0,0]**2+rbm.bv[0,1]**2)])
        weights.append([numx.sqrt(numx.sum(rbm.w[:,0]**2)),numx.sqrt(numx.sum(rbm.w[:,1]**2)),numx.sqrt(numx.sum(rbm.w[:,2]**2)),numx.sqrt(numx.sum(rbm.w[:,3]**2))])
        m = numx.mean(train_data[b:(b+batch_size),:],axis = 0)
        means.append([m[0],m[1]])
    #print epsilon[0]
    #epsilon[0] *= 0.98
    #epsilon[1] *= 0.98
    #epsilon[2] *= 0.98
    #epsilon[3] *= 0.98

    # Calculate Log likelihood and reconstruction error
    RE_train = numx.mean(ESTIMATOR.reconstruction_error(rbm, train_data))
    RE_test = numx.mean(ESTIMATOR.reconstruction_error(rbm, test_data))
    Z = ESTIMATOR.partition_function_factorize_h(rbm, batchsize_exponent=h1)
    LL_train = numx.mean(ESTIMATOR.log_likelihood_v(rbm, Z , train_data))
    LL_test = numx.mean(ESTIMATOR.log_likelihood_v(rbm, Z , test_data))
    print '%5d \t%0.5f \t%0.5f \t%0.5f \t%0.5f' % (epoch, RE_train, RE_test, LL_train,  LL_test)

# Calculate partition function and its AIS approximation
Z = ESTIMATOR.partition_function_factorize_h(rbm, batchsize_exponent=h1)
Z_AIS = ESTIMATOR.annealed_importance_sampling(rbm, num_chains=100, k=1, betas=1000,status=False)[0]

# Calculate and print LL
print ""
print "\nTrue log partition: ", Z," ( LL_train: ",numx.mean(ESTIMATOR.log_likelihood_v(rbm, Z , train_data)),",","LL_test: ",numx.mean(ESTIMATOR.log_likelihood_v(rbm, Z , test_data))," )"
print "\nAIS  log partition: ", Z_AIS," ( LL_train: ",numx.mean(ESTIMATOR.log_likelihood_v(rbm, Z_AIS , train_data)),",","LL_test: ",numx.mean(ESTIMATOR.log_likelihood_v(rbm, Z_AIS , test_data))," )"
print ""
# Print parameter
print '\nWeigths:\n',rbm.w
print 'Visible bias:\n',rbm.bv
print 'Hidden bias:\n',rbm.bh
print 'Sigmas:\n',rbm.sigma
print 

# Calculate P(h) wich are the scaling factors of the Gaussian components
h_i = numx.zeros((1,h1*h2))
print 'P(h_0)',numx.exp(rbm.log_probability_h(Z,h_i))
for i in range(h1*h2):
    h_i = numx.zeros((1,h1*h2))
    h_i[0,i]=1
    print 'P(h',(i+1),')',numx.exp(rbm.log_probability_h(Z,h_i))

# Display results
# create a new figure of size 5x5
VISUALIZATION.figure(0, figsize = [5,5])
VISUALIZATION.title("P(x)")
# plot the data
#for i in range(1,len(biases)):
#    VISUALIZATION.arrow(biases[i-1][0],biases[i-1][1], biases[i][0],biases[i][1], color='black', width=0.02, length_includes_head=False,
#          head_width=0.02)
VISUALIZATION.plot_2d_data(whiteData, alpha = 0.8,color = 'lightgray')
# plot weights
VISUALIZATION.plot_2d_weights(rbm.w, rbm.bv,1,'black','blue')
# pass our P(x) as function to plotting function
VISUALIZATION.plot_2d_contour(lambda v: numx.exp(rbm.log_probability_v(Z, v)))
# No inconsistent scaling
VISUALIZATION.axis('equal')
# Set size of the plot
VISUALIZATION.axis([-5,5,-5,5])

# Do the sam efor the LOG-Plot
# create a new figure of size 5x5
VISUALIZATION.figure(1, figsize = [5,5])
VISUALIZATION.title("Ln( P(x) )")
# plot the data
VISUALIZATION.plot_2d_data(whiteData, alpha = 0.8, color = 'lightgray')
# plot weights
VISUALIZATION.plot_2d_weights(rbm.w, rbm.bv,1,'black','blue')
# pass our P(x) as function to plotting function
VISUALIZATION.plot_2d_contour(lambda v: rbm.log_probability_v(Z, v))
# No inconsistent scaling
VISUALIZATION.axis('equal')
# Set size of the plot
VISUALIZATION.axis([-5,5,-5,5])

VISUALIZATION.show()


VISUALIZATION.figure(2, figsize = [5,5])
VISUALIZATION.plot(numx.array(weights)[:,0])
VISUALIZATION.figure(3, figsize = [5,5])
VISUALIZATION.plot(numx.array(weights)[:,1])
VISUALIZATION.figure(4, figsize = [5,5])
VISUALIZATION.plot(numx.array(weights)[:,2])
VISUALIZATION.figure(5, figsize = [5,5])
VISUALIZATION.plot(numx.array(weights)[:,3])

VISUALIZATION.figure(6, figsize = [5,5])
VISUALIZATION.plot(biases)


VISUALIZATION.show()
