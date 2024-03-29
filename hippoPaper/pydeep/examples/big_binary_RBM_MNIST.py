''' Example using a big BB-RBMs on the MNIST handwritten digit database.

    :Version:
        1.1.0

    :Date:
        24.04.2017

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2017 Jan Melchior

        This file is part of the Python library PyDeep.

        PyDeep is free software: you can redistribute it and/or modify
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
import pydeep.rbm.model as model
import pydeep.rbm.trainer as trainer
import pydeep.rbm.estimator as estimator

import pydeep.misc.io as io
import pydeep.misc.visualization as vis
import pydeep.misc.measuring as mea

# Set random seed (optional)
numx.random.seed(42)

# Input and hidden dimensionality
v1 = v2 = 28
h1 = 25
h2 = 20

# Load data , get it from 'deeplearning.net/data/mnist/mnist.pkl.gz'
train_data = io.load_mnist("../../../data/mnist.pkl.gz", True)[0]

# Training paramters
batch_size = 100
epochs = 200
rbm = io.load_object("mnist500.rbm")

# Create trainer and model
rbm = model.BinaryBinaryRBM(number_visibles=v1 * v2,
                            number_hiddens=h1 * h2,
                            data=train_data)
trainer_pcd = trainer.PCD(rbm, batch_size)

# Measuring time
measurer = mea.Stopwatch()

# Train model
print('Training')
print('Epoch\t\tRecon. Error\tLog likelihood \tExpected End-Time')
for epoch in range(1, epochs + 1):

    # Shuffle training samples (optional)
    train_data = numx.random.permutation(train_data)

    # Loop over all batches
    for b in range(0, train_data.shape[0], batch_size):
        batch = train_data[b:b + batch_size, :]
        trainer_pcd.train(data=batch, epsilon=0.01)

    # Calculate reconstruction error and expected end time every 10th epoch
    if epoch % 10 == 0:
        RE = numx.mean(estimator.reconstruction_error(rbm, train_data))
        print('{}\t\t{:.4f}\t\t\t{}'.format(epoch, RE, measurer.get_expected_end_time(epoch, epochs)))
    else:
        print(epoch)

# Save the model
io.save_object(rbm,"mnist500.rbm")

# Stop time measurement
measurer.end()

# Print end time
print("End-time: \t{}".format(measurer.get_end_time()))
print("Training time:\t{}".format(measurer.get_interval()))

# Approximate partition function using AIS for lower bound approximiation
Z = estimator.annealed_importance_sampling(rbm)[0]
print("AIS Partition: {} (LL: {})".format(Z, numx.mean(estimator.log_likelihood_v(rbm, Z, train_data))))

# Approximate partition function using reverse AIS for upper bound approximiation
Z = estimator.reverse_annealed_importance_sampling(rbm)[0]
print("reverse AIS Partition: {} (LL: {})".format(Z, numx.mean(estimator.log_likelihood_v(rbm, Z, train_data))))

# Reorder RBM features by average activity decreasingly
reordered_rbm = vis.reorder_filter_by_hidden_activation(rbm, train_data)

# Display RBM parameters
vis.imshow_standard_rbm_parameters(reordered_rbm, v1, v2, h1, h2)

# Sample some steps and show results
samples = vis.generate_samples(rbm, train_data[0:30], 30, 1, v1, v2, False, None)
vis.imshow_matrix(samples, 'Samples')

# Display results
vis.show()
