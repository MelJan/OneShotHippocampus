''' Example using GB-RBMs on a natural image patches.

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

import pydeep.base.numpyextension as numxext
import pydeep.misc.io as io
import pydeep.preprocessing as pre
import pydeep.misc.visualization as vis
import pydeep.rbm.estimator as estimator
import pydeep.rbm.model as model
import pydeep.rbm.trainer as trainer

data = numx.random.permutation(io.load_natural_image_patches('../../../data/NaturalImage.mat'))
#data = IO.load_cifar('/home/jan/PycharmProjects/data/2',True)

data = pre.remove_rows_means(data)

h1 = 14
h2 = 14
v1 = 14
v2 = 14

zca = pre.ZCA(v1 * v2)
zca.train(data)
data = zca.project(data)

train_data = data[0:40000]
test_data = data[40000:70000]

restrict = 0.01*numx.max(numxext.get_norms(train_data, axis = 1))

batch_size = 100

max_epochs = 200
step = 10
eps = 0.1

numx.random.seed(42)

# Train model
print 'Training'
print 'Epoch\tRE train_images\tRE test \tLL train_images\tLL test '
# Create model
rbm = model.GaussianBinaryVarianceRBM(number_visibles = v1*v2,
                                      number_hiddens = h1*h2,
                                      data=train_data,
                                      initial_weights='AUTO',
                                      initial_visible_bias=0,
                                      initial_hidden_bias=0,
                                      initial_sigma=1.0,
                                      initial_visible_offsets=0.0,
                                      initial_hidden_offsets=0.0,
                                      dtype=numx.float64)
rbm.bh = -(numxext.get_norms(rbm.w+rbm.bv.T, axis = 0)-numxext.get_norms(rbm.bv, axis = None))/2.0+numx.log(0.01)
rbm.bh = rbm.bh.reshape(1,h1*h2)
k = 1
trainer = trainer.CD(rbm)
for epoch in range(0,max_epochs+1,1) :
    train_data = numx.random.permutation(train_data)

    if epoch % step == 0:
        RE_train = numx.mean(estimator.reconstruction_error(rbm, train_data))
        RE_test = numx.mean(estimator.reconstruction_error(rbm, test_data))
        print '%5d \t%0.5f \t%0.5f' % (epoch, RE_train, RE_test)

    for b in range(0,train_data.shape[0],batch_size):
        trainer.train(data = train_data[b:(b+batch_size),:],
                      num_epochs=1,
                      epsilon=[eps,eps,eps,eps*0.1],
                      k=k,
                      momentum=0.0,
                      reg_l1norm=0.0,
                      reg_l2norm=0.0,
                      reg_sparseness = 0,
                      desired_sparseness=None,
                      update_visible_offsets=0,
                      update_hidden_offsets=0.01,
                      offset_typ = '0D',
                      restrict_gradient=restrict,
                      restriction_norm='Cols',
                      use_hidden_states=False,
                      use_centered_gradient=False)


RE_train = numx.mean(estimator.reconstruction_error(rbm, train_data))
RE_test = numx.mean(estimator.reconstruction_error(rbm, test_data))
Z = estimator.annealed_importance_sampling(rbm)[0]
LL_train = numx.mean(estimator.log_likelihood_v(rbm, Z , train_data))
LL_test = numx.mean(estimator.log_likelihood_v(rbm, Z , test_data))
print '%5d \t%0.5f \t%0.5f \t%0.5f \t%0.5f' % (max_epochs, RE_train, RE_test, LL_train, LL_test)

Z = estimator.reverse_annealed_importance_sampling(rbm)[0]
LL_train = numx.mean(estimator.log_likelihood_v(rbm, Z , train_data))
LL_test = numx.mean(estimator.log_likelihood_v(rbm, Z , test_data))
print '%5d \t%0.5f \t%0.5f \t%0.5f \t%0.5f' % (max_epochs, RE_train, RE_test, LL_train, LL_test)

# Prepare results
rbmReordered = vis.reorder_filter_by_hidden_activation(rbm, train_data)
vis.imshow_standard_rbm_parameters(rbmReordered, v1,v2,h1, h2)
samples = vis.generate_samples(rbm, train_data[0:30], 30, 1, v1, v2, False, None)
vis.imshow_matrix(samples,'Samples')



vis.show()