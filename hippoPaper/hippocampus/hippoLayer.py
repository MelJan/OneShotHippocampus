""" Hebb-layers.

    :Version:
        2.0.0

    :Date:
        04.07.2018

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2018 Jan Melchior

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
"""
import numpy as np
from pydeep.base.numpyextension import restrict_norms
from hebbActivationFunction import HebbSigmoid,HebbStep
from pydeep.base.costfunction import SquaredError, CrossEntropyError
from pydeep.base.corruptor import *

import pydeep.fnn.layer as layer
import pydeep.fnn.model as model
import pydeep.fnn.trainer as trainer
from dataProvider import *


class HeteroAssociator(object):
    """ Heteroassociative single layer network.

    """

    def __init__(self,
                 input_dim,
                 input_mean,
                 output_dim,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 connection_matrix=None
                 ):
        """ Constructor for a hetero associator network.

        :param input_dim: Input dimensionality.
        :type input_dim: int

        :param input_mean: Average input activity of the input [0,1].
        :type input_mean: float

        :param output_dim: Output dimensionality.
        :type output_dim: int

        :param act: Activation function.
        :type act: hebbActivationFunction.

        :param cost: Cost function i.e. SquaredError
        :type cost: pydeep.base.costfunction.

        :param connection_matrix: Connectivity matrix.
        :type input_dim: 2D numpy array

        """
        # Store provided parameters
        self.input_dim = input_dim
        self.input_mean = input_mean
        self.output_dim = output_dim
        self.cost = cost
        self.act = act
        self.model = model.Model([layer.FullConnLayer(input_dim=input_dim,
                                                      output_dim=output_dim,
                                                      activation_function=self.act,
                                                      initial_weights='AUTO',
                                                      initial_bias=0.0,
                                                      initial_offset=input_mean,
                                                      connections=connection_matrix,
                                                      dtype=np.float64)])
        self.trainer = trainer.GDTrainer(self.model)

    def store_datapoint(self,
                        input_data,
                        output_data,
                        epochs=1,
                        epsilon=1.0,
                        update_offsets=0.0,
                        momentum=0.0,
                        l1norm=0.0,
                        l2norm=0.0,
                        corruptor=None
                        ):
        """ Stroes a new datapoint, online learning.

        :param input_data: Input datapoint(s)
        :type input_data: numpy array

        :param output_data: Output datapoint(s)
        :type output_data: numpy array

        :param epochs: Number of epochs.
        :type epochs: int

        :param epsilon: Learning rate.
        :type epsilon: float

        :param update_offsets: Offset shifting term 0 < shift < 1.0.
        :type update_offsets: float

        :param momentum: Momentum term 0 < momentum < 1.0.
        :type momentum: float

        :param l1norm: L1 Norm decay usually around 0.0001.
        :type l1norm: float

        :param l2norm: L2 Norm decay usually around 0.0001.
        :type l2norm: float

        :param corruptor:
        :type corruptor: pydeep.base.corruptor

        """
        for e in range(epochs):

            self.trainer.train(data=input_data,
                               labels=[output_data],
                               costs=[self.cost],
                               reg_costs=[1],
                               momentum=[momentum],
                               epsilon=[epsilon],
                               update_offsets=[update_offsets],
                               corruptor=[corruptor, None],
                               reg_L1Norm=[l1norm],
                               reg_L2Norm=[l2norm],
                               reg_sparseness=[0.0],
                               desired_sparseness=[0.0],
                               costs_sparseness=[None, None],
                               restrict_gradient=[0.0],
                               restriction_norm='Mat'
                               )

            """ Proof of concept, WARNING no easy switch between backprop and hebb!!! (BUT IT IS FASTER) """
            """
            h = self.model.forward_propagate(input_data)
            mu = self.model.layers[0].offset
            self.model.layers[0].weights -= epsilon/input_data.shape[0]*numx.dot((input_data-mu).T,h-output_data)
            self.model.layers[0].bias -= epsilon*numx.mean(h-output_data,axis=0).reshape(1,self.output_dim)
            """
            """
            if input_data.shape[0] > 1:
                print "EEEEEEEEEEEEERRRRRRRRRRRROR"
            self.model.layers[0].weights += np.dot((input_data - self.model.layers[0].offset).T, output_data-0.35)-0.01*self.model.layers[0].weights
            """

    def calculate_output(self, input_data, binarize_output=False):
        """ Calculates the output of the network.

        :param input_data: Input data.
        :type input_data: numpy array.

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: Output of the network.
        :rtype: numpy array

        """
        if binarize_output:
            return np.float64(self.model.forward_propagate(input_data) > 0.5)
        else:
            return self.model.forward_propagate(input_data)


class AutoAssociator(object):

    def __init__(self,
                 input_dim,
                 input_mean,
                 output_dim,
                 output_mean,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 connection_matrix=None
                 ):
        """ Constructor for Auto Associator network.

        :param input_dim: Input dimensionality.
        :type input_dim: int.

        :param input_mean: Average activity of the input [0,1].
        :type input_mean: float.

        :param output_dim: Output dimensionality.
        :type output_dim: int.

        :param output_mean: Average activity of the output [0,1].
        :type output_mean: float.

        :param act: If True as parse hidden representation is learned.
        :type act: bool.

        :param act: Activation function.
        :type act: hebbActivationFunction.

        :param cost: Cost function i.e. SquaredError
        :type cost: pydeep.base.costfunction.

        :param connection_matrix: Connectivity matrix.
        :type input_dim: 2D numpy array.

        """
        # Store provided parameters
        self.input_dim = input_dim
        self.input_activity = input_mean
        self.output_dim = output_dim
        self.output_activity = output_mean
        self.cost = cost
        self.act = act

        layer1 = layer.FullConnLayer(input_dim=input_dim,
                                     output_dim=output_dim,
                                     activation_function=self.act,
                                     initial_weights='AUTO',
                                     initial_bias=0.0,
                                     initial_offset=input_mean,
                                     connections=connection_matrix,
                                     dtype=np.float64)
        if connection_matrix is None:
            layer2 = layer.FullConnLayer(input_dim=output_dim,
                                         output_dim=input_dim,
                                         activation_function=self.act,
                                         initial_weights=layer1.weights.T,
                                         initial_bias=0.0,
                                         initial_offset=output_mean,
                                         connections=None,
                                         dtype=np.float64)
        else:
            layer2 = layer.FullConnLayer(input_dim=output_dim,
                                         output_dim=input_dim,
                                         activation_function=self.act,
                                         initial_weights=layer1.weights.T,
                                         initial_bias=0.0,
                                         initial_offset=output_mean,
                                         connections=connection_matrix.T,
                                         dtype=np.float64)
        self.model = model.Model([layer1, layer2])
        self.trainer = trainer.GDTrainer(self.model)

    def train(self,
              epochs=1000,
              epsilon=1.0,
              update_offsets=0.0,
              momentum=0.0,
              l1norm=0.0,
              l2norm=0.0,
              corruptor=None
              ):
        """ Trains the network on random data.

        :param epochs: Number of epochs.
        :type epochs: int

        :param epsilon: Learning rate.
        :type epsilon: float

        :param update_offsets: Offset shifting term 0 < shift < 1.0.
        :type update_offsets: float

        :param momentum: Momentum term 0 < momentum < 1.0.
        :type momentum: float

        :param l1norm: L1 Norm decay usually around 0.0001.
        :type l1norm: float

        :param l2norm: L2 Norm decay usually around 0.0001.
        :type l2norm: float

        :param corruptor: list of corruptors or None.
        :type corruptor: list of pydeep.base.corruptor or None

        """
        for e in range(epochs):

            dataset = generate_binary_random_sequence(10, self.input_dim, np.int32(self.input_dim *
                                                                                    self.input_activity))
            self.trainer.train(data=dataset,
                               labels=[None, dataset],
                               costs=[None, self.cost],
                               reg_costs=[0, 1],
                               momentum=[momentum, momentum],
                               epsilon=[0, epsilon],
                               update_offsets=[update_offsets, update_offsets],
                               corruptor=corruptor,
                               reg_L1Norm=[l1norm, l1norm],
                               reg_L2Norm=[l2norm, l2norm],
                               reg_sparseness=[0.0, 0.0],
                               desired_sparseness=[0, 0],
                               costs_sparseness=[None ,None],
                               restrict_gradient=[0,0],
                               restriction_norm='Mat'
                               )
            self.model.layers[0].bias -= epsilon *0.5* np.mean(self.model.layers[0].temp_a - self.output_activity,
                                                               axis=0).reshape(1,self.output_dim)
            #self.model.layers[0].bias -= epsilon *0.5* np.mean(np.mean(self.model.layers[0].temp_a,axis=1) - self.output_activity,
            #                                                   axis=0)

            """
            dataset = generate_binary_random_sequence(1000, self.input_dim, np.int32(self.input_dim *
                                                                                    self.input_activity))

            self.trainer.train(data=dataset,
                               labels=[None, dataset],
                               costs=[None, self.cost],
                               reg_costs=[0, 1],
                               momentum=[momentum, momentum],
                               epsilon=[epsilon, epsilon],
                               update_offsets=[update_offsets, update_offsets],
                               corruptor=corruptor,
                               reg_L1Norm=[l1norm, l1norm],
                               reg_L2Norm=[l2norm, l2norm],
                               reg_sparseness=[1.0, 0.0],
                               desired_sparseness=[self.output_activity, 0],
                               costs_sparseness=[self.cost ,None],
                               restrict_gradient=[100,100],
                               restriction_norm='Mat'
                               )
            """
            """ Proof of concept,
                USE THIS IF BACKPROP IS NEEDED! no easy switch between backprop and Hebb!!! (BUT IT IS FASTER) """
            """
            x_tilde = self.experiments.forward_propagate(dataset)
            h = self.experiments.layers[0].temp_a
            mu = self.experiments.layers[0].offset
            weights =self.experiments.layers[0].weights
            grad_b = numx.dot(x_tilde-dataset,weights)+(h-self.output_activity)
            grad_w = restrict_norms(epsilon/100.0*numx.dot((dataset-mu).T,grad_b), 10.0, None)
            self.experiments.layers[0].weights -= grad_w
            self.experiments.layers[0].bias -= epsilon*numx.mean(grad_b,axis=0).reshape(1,self.output_dim)
            """
            # Also update V= W^T and c, makes no siginifacent diff
            #grad_c = x_tilde - dataset
            #grad_v = restrict_norms(epsilon / 100.0 * numx.dot((h - lam).T, grad_c), 10.0, None)
            # lam = self.experiments.layers[1].offset
            #self.experiments.layers[1].weights -= grad_v
            #self.experiments.layers[01].bias -= epsilon*numx.mean(grad_c,axis=0).reshape(1,self.input_dim)

            print "SPARSITY\t", "RECONSTRUCTION ERROR"
            dataset = generate_binary_random_sequence(100, self.input_dim, np.int32(self.input_dim *
                                                                                  self.input_activity))
            print e,np.mean(np.abs(self.calculate_output(dataset, False))), np.mean(np.abs(
                self.calculate_reconstruct(dataset, False) - dataset))

    def calculate_reconstruct(self, input_data, binarize_output=False):
        """ Calculates the reconstruction of the network.

            :param input_data: Input data.
            :type input_data: numpy array.

            :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
            :type binarize_output: bool

            :return: Reconstruction of the network.
            :rtype: numpy array

        """
        if binarize_output:
            return np.float64(self.model.forward_propagate(input_data) > 0.5)
        else:
            return self.model.forward_propagate(input_data)

    def calculate_output(self, input_data, binarize_output=False):
        """ Calculates the output of the network.

        :param input_data: Input data.
        :type input_data: numpy array.

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: Output of the network.
        :rtype: numpy array.

        """
        if binarize_output:
            return np.float64(self.model.layers[0].forward_propagate(input_data) > 0.5)
        else:
            return self.model.layers[0].forward_propagate(input_data)

    def calculate_input(self, output_data, binarize_output=False):
        """ Calculates the output of the network.

        :param output_data: Output data.
        :type output_data: numpy array.

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: Input of the network.
        :rtype: numpy array.

        """
        if binarize_output:
            return np.float64(self.model.layers[1].forward_propagate(output_data) > 0.5)
        else:
            return self.model.layers[1].forward_propagate(output_data)


class PredictiveAssociator(object):
    """ Hebbian Markovian-predictor, with warp around i.e. pattern 1 follows pattern N.

    """

    def __init__(self,
                 sequence,
                 completion_loops=0,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 store_training_sequence=True
                 ):
        """
        Initializes the neural network.

        :param sequence: Sequence to be stored.
        :type sequence: numpy array

        :param completion_loops: Number of pattern completion loops.
        :type completion_loops: int

        :param act: Activation function, if single layer + Squared error + HebbSigmoid -> generlized Hebb learning.
        :type sequence: numpy array

        :param cost: Cost function, if single layer + Squared error + HebbSigmoid -> generlized Hebb learning.
        :type cost: pydeep.base.costfunction

        :param store_training_sequence: If True the original training sequence is stored for comparision.
        :type store_training_sequence: bool

        """
        # Store dimensions
        self.seq_dim = sequence.shape[1]
        self.sequence_length = sequence.shape[0]
        self.completion_loops = completion_loops
        self.cost = cost
        self.act = act
        hidden_layer = None
        # Create 1 layer network
        if hidden_layer is None:
            layer1_forward = layer.FullConnLayer(input_dim=self.seq_dim,
                                                 output_dim=self.seq_dim,
                                                 activation_function=self.act,
                                                 initial_weights='AUTO',
                                                 initial_bias=0.0,
                                                 initial_offset=np.mean(sequence, axis=0).reshape(1, sequence.shape[1]),
                                                 connections=None,
                                                 dtype=np.float64)
            self.model_forward = model.Model([layer1_forward])
            layer1_backward = layer.FullConnLayer(input_dim=self.seq_dim,
                                                  output_dim=self.seq_dim,
                                                  activation_function=self.act,
                                                  initial_weights='AUTO',
                                                  initial_bias=0.0,
                                                  initial_offset=np.mean(sequence, axis=0).reshape(1, sequence.shape[1]),
                                                  connections=None,
                                                  dtype=np.float64)
            self.model_backward = model.Model([layer1_backward])
        else:
            layer1_forward = layer.FullConnLayer(input_dim=self.seq_dim,
                                                 output_dim=self.seq_dim * 2,
                                                 activation_function=self.act,
                                                 initial_weights='AUTO',
                                                 initial_bias=0.0,
                                                 initial_offset=np.mean(sequence, axis=0).reshape(1, sequence.shape[1]),
                                                 connections=None,
                                                 dtype=np.float64)
            layer2_forward = layer.FullConnLayer(input_dim=self.seq_dim * 2,
                                                 output_dim=self.seq_dim,
                                                 activation_function=self.act,
                                                 initial_weights='AUTO',
                                                 initial_bias=0.0,
                                                 initial_offset=0,
                                                 connections=None,
                                                 dtype=np.float64)
            self.model_forward = model.Model([layer1_forward, layer2_forward])

            layer1_backward = layer.FullConnLayer(input_dim=self.seq_dim,
                                                  output_dim=self.seq_dim * 2,
                                                  activation_function=self.act,
                                                  initial_weights='AUTO',
                                                  initial_bias=0.0,
                                                  initial_offset=np.mean(sequence, axis=0).reshape(1,
                                                                                                   sequence.shape[1]),
                                                  connections=None,
                                                  dtype=np.float64)
            layer2_backward = layer.FullConnLayer(input_dim=self.seq_dim * 2,
                                                  output_dim=self.seq_dim,
                                                  activation_function=self.act,
                                                  initial_weights='AUTO',
                                                  initial_bias=0.0,
                                                  initial_offset=0,
                                                  connections=None,
                                                  dtype=np.float64)
            self.model_backward = model.Model([layer1_backward, layer2_backward])

        self.trainer_forward = trainer.GDTrainer(self.model_forward)
        self.trainer_backward = trainer.GDTrainer(self.model_backward)

        # Current pattern, used to return the next pattern by get_next_pattern
        # i.e. It represents the intrinsic state of the network
        self.current_state = None

        # If True the training sequence is not deleted after training, just for the case you want to keep it e.g. save
        # all to gard disk and want to recover the exact sequence later
        self.store_training_sequence = store_training_sequence

        # Store and if desired binarize training sequence
        self._training_sequence = sequence

    def train(self,
              epsilon=0.1,
              disturbance=0.05,
              max_epochs=1000,
              batch_size=1):
        """
        :param epsilon: Learning rate.
        :type epsilon: float

        :param disturbance: Percentage of noise added to the input.
        :type disturbance: float

        :param max_epochs: Maximum number of epochs for training.
        :type max_epochs: int

        :param batch_size: Batch size used during training.
        :type batch_size: int

        """
        # Roll sequence to get target output for the network.
        sequence_next = np.roll(self._training_sequence, -1, 0)
        corr = None
        if disturbance > 0:
            corr = RandomPermutation(disturbance)
        for epoch in range(1, max_epochs + 1):
            for b in range(0, self._training_sequence.shape[0], batch_size):
                if self.model_backward.num_layers == 1:

                    self.trainer_forward.train(data=self._training_sequence[b:b + batch_size:],
                                               labels=[sequence_next[b:b + batch_size, :]],
                                               costs=[self.cost],
                                               reg_costs=[1],
                                               momentum=[0],
                                               epsilon=[epsilon],
                                               update_offsets=[0],
                                               corruptor=[corr, None],
                                               reg_L1Norm=[0],
                                               reg_L2Norm=[0],
                                               reg_sparseness=[0],
                                               desired_sparseness=[0],
                                               costs_sparseness=[None],
                                               restrict_gradient=[0.0],
                                               restriction_norm='Mat')
                    self.trainer_backward.train(data=sequence_next[b:b + batch_size, :],
                                                labels=[self._training_sequence[b:b + batch_size:]],
                                                costs=[self.cost],
                                                reg_costs=[1],
                                                momentum=[0],
                                                epsilon=[epsilon],
                                                update_offsets=[0],
                                                corruptor=[corr, None],
                                                reg_L1Norm=[0],
                                                reg_L2Norm=[0],
                                                reg_sparseness=[0],
                                                desired_sparseness=[0],
                                                costs_sparseness=[None],
                                                restrict_gradient=[0.0],
                                                restriction_norm='Mat')

                    """ Proof of concept, WARNING no easy switch between backprop and Hebb!!!
                        Only works with one hidden layer (BUT IT IS FASTER)
                    """
                    """
                    input_data = self._training_sequence[b:b + batch_size:]
                    output_data = sequence_next[b:b + batch_size, :]
                    h = self.model_forward.forward_propagate(input_data)
                    mu = self.model_forward.layers[0].offset
                    self.model_forward.layers[0].weights -= epsilon / input_data.shape[0] * numx.dot((input_data - mu).T,
                                                                                             h - output_data)
                    self.model_forward.layers[0].bias -= epsilon * numx.mean(h - output_data, axis=0)

                    input_data = sequence_next[b:b + batch_size, :]
                    output_data = self._training_sequence[b:b + batch_size:]
                    h = self.model_backward.forward_propagate(input_data)
                    mu = self.model_backward.layers[0].offset
                    self.model_backward.layers[0].weights -= epsilon / input_data.shape[0] * numx.dot((input_data - mu).T,
                                                                                             h - output_data)
                    self.model_backward.layers[0].bias -= epsilon * numx.mean(h - output_data, axis=0)
                    """

                else:
                    self.trainer_forward.train(data=self._training_sequence[b:b + batch_size:],
                                               labels=[None, sequence_next[b:b + batch_size, :]],
                                               costs=[None, self.cost],
                                               reg_costs=[0, 1],
                                               momentum=[0, 0],
                                               epsilon=[epsilon, epsilon],
                                               update_offsets=[0, 0],
                                               corruptor=[corr, None, None],
                                               reg_L1Norm=[0, 0],
                                               reg_L2Norm=[0, 0],
                                               reg_sparseness=[0, 0],
                                               desired_sparseness=[0, 0],
                                               costs_sparseness=[None, None],
                                               restrict_gradient=0.0,
                                               restriction_norm='Mat')
                    self.trainer_backward.train(data=sequence_next[b:b + batch_size, :],
                                                labels=[None, self._training_sequence[b:b + batch_size:]],
                                                costs=[None, self.cost],
                                                reg_costs=[0, 1],
                                                momentum=[0, 0],
                                                epsilon=[epsilon, epsilon],
                                                update_offsets=[0, 0],
                                                corruptor=[corr, None, None],
                                                reg_L1Norm=[0, 0],
                                                reg_L2Norm=[0, 0],
                                                reg_sparseness=[0, 0],
                                                desired_sparseness=[0, 0],
                                                costs_sparseness=[None, None],
                                                restrict_gradient=0.0,
                                                restriction_norm='Mat')
            start = self._training_sequence[0]
            for i in range(self.sequence_length):
                start = self.calculate_output(start)
            print epoch,np.mean(np.abs(start - self._training_sequence[0]))
            self.current_state = np.atleast_2d(self._training_sequence[0])

        # If desired store original sequence
        if not self.store_training_sequence:
            self._training_sequence = None

        print "Performance CA3 forward"
        print "Error - Full loop Error"
        print np.mean(np.abs(self.calculate_output(np.roll(self._training_sequence, 1, 0)) - self._training_sequence)),

        print "Performance CA3 forward 2"
        print "Error - Full loop Error"
        print np.mean(np.abs(self.calculate_output(self.calculate_output(np.roll(self._training_sequence, 2, 0))) - self._training_sequence)),

        print "Performance CA3 forward 3"
        print "Error - Full loop Error"
        print np.mean(np.abs(self.calculate_output(self.calculate_output(
            self.calculate_output(np.roll(self._training_sequence, 3, 0)))) - self._training_sequence)),

        print "Performance CA3 forward 4"
        print "Error - Full loop Error"
        print np.mean(np.abs(self.calculate_output(self.calculate_output(self.calculate_output(
            self.calculate_output(np.roll(self._training_sequence, 4, 0))))) - self._training_sequence)),

        start = self._training_sequence[0]
        for i in range(self.sequence_length):
            start = self.calculate_output(start)
        print np.mean(np.abs(start - self._training_sequence[0]))
        print "Performance CA3 backward"
        print "Error - Full loop Error"
        print np.mean(np.abs(self.calculate_input(np.roll(self._training_sequence, -1, 0)) - self._training_sequence)),
        start = self._training_sequence[0]
        for i in range(self.sequence_length):
            start = self.calculate_input(start)
        print np.mean(np.abs(start - self._training_sequence[0]))

    def calculate_full_loop_reconstruction(self, dataset, forward=True, repeats=1, binarize_output=False):
        """ Calculated the full-loop reconstruction.
            Full loop means starting from a pattern it predicts the next, this is taken to predict the next, ... until
            you reach the prediction for the pattern we started from. This is done in parallel for all patterns in the
            given batch.

        :param dataset: Data samples
        :type dataset: numpy array

        :param repeats: Number of loops through the data
        :type repeats: int

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: Reconstruction
        :rtype: numpy array

        """
        # Calculate as full loop reconstruction for all given patterns
        result = None
        next_pattern = dataset
        for _ in range(repeats):
            for i in range(self.sequence_length):
                if forward:
                    next_pattern = self.calculate_output(next_pattern, binarize_output)
                else:
                    next_pattern = self.calculate_input(next_pattern, binarize_output)
                if result is None:
                    result = next_pattern
                else:
                    result = np.vstack((result, next_pattern))
        return result

    def set_current_state(self, offset_pattern, binarize_output=False):
        """ Sets/Overwrites the current pattern/state of the network.

        :param offset_pattern: new pattern/state.
        :type offset_pattern: numpy array

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        """
        self.current_state = offset_pattern
        if binarize_output:
            self.current_state = np.float64(self.current_state > 0.5)

    def get_current_state(self, binarize_output=False):
        """ Returns the current pattern/state of the network.

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: current pattern/state.
        :rtype: numpy array

        """
        if binarize_output:
            return np.float64(self.current_state > 0.5)
        return self.current_state

    def calculate_output(self, pattern, binarize_output=False):
        """ Returns the output for the network for a given pattern.
            The current pattern is NOT updated!!! see get_next_state to do so.

        :param pattern: Input pattern.
        :type pattern: numpy array

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: Output/prediction.
        :rtype: numpy array

        """
        output = self.model_forward.forward_propagate(pattern)
        if binarize_output:
            output = np.float64(output > 0.5)
        for _ in range(self.completion_loops):
            output = self.model_backward.forward_propagate(output)
            if binarize_output:
                output = np.float64(output > 0.5)
            output = self.model_forward.forward_propagate(output)
            if binarize_output:
                output = np.float64(output > 0.5)
        return output

    def get_next_state(self, binarize_output=False):
        """ Returns and UPDATES the current pattern/state.
            i.e. Current is set to the next pattern.

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: next pattern/state.
        :rtype: numpy array

        """
        self.current_state = self.calculate_output(self.current_state, binarize_output)
        return self.get_current_state(binarize_output)

    def calculate_input(self, pattern, binarize_output=False):
        """ Returns the output for the network for a given pattern.
            The current pattern is NOT updated!!! see get_next_state to do so.

        :param pattern: Input pattern.
        :type pattern: numpy array

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: Output/prediction.
        :rtype: numpy array

        """
        output = self.model_backward.forward_propagate(pattern)
        if binarize_output:
            output = np.float64(output > 0.5)
        for _ in range(self.completion_loops):
            output = self.model_forward.forward_propagate(output)
            if binarize_output:
                output = np.float64(output > 0.5)
            output = self.model_backward.forward_propagate(output)
            if binarize_output:
                output = np.float64(output > 0.5)
        return output

    def get_previous_state(self, binarize_output=False):
        """ Returns and UPDATES the current pattern/state.
            i.e. Current is set to the next pattern.

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: next pattern/state.
        :rtype: numpy array

        """
        self.current_state = self.calculate_input(self.current_state, binarize_output)
        return self.get_current_state(binarize_output)


'''
class PredictiveAssociator(object):
    """ Hebbian Markovian-predictor, with warp around i.e. pattern 1 follows pattern N.

    """

    def __init__(self,
                 sequence,
                 hidden_dim,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 store_training_sequence=True
                 ):
        """
        Initializes the neural network.

        :param sequence: Sequence to be stored.
        :type sequence: numpy array

        :param hidden_dim: Number of hidden units in the artificial neural network.
        :type hidden_dim: int

        :param act: Activation function, if single layer + Squared error + HebbSigmoid -> generlized Hebb learning.
        :type sequence: numpy array

        :param cost: Cost function, if single layer + Squared error + HebbSigmoid -> generlized Hebb learning.
        :type cost: pydeep.base.costfunction

        :param store_training_sequence: If True the original training sequence is stored for comparision.
        :type store_training_sequence: bool

        """
        # Store dimensions
        self.seq_dim = sequence.shape[1]
        self.sequence_length = sequence.shape[0]
        self.hidden_dim = hidden_dim

        self.cost = cost

        if self.hidden_dim is None or self.hidden_dim == 0:
            # Create 1 layer network
            layer1 = layer.FullConnLayer(input_dim=self.seq_dim,
                                         output_dim=self.seq_dim,
                                         activation_function=act,
                                         initial_weights='AUTO',
                                         initial_bias=0.0,
                                         initial_offset=np.mean(sequence, axis=0).reshape(1, sequence.shape[1]),
                                         connections=None,
                                         dtype=np.float64)
            self.experiments = experiments.Model([layer1])
        else:
            # Create 2 layer network
            layer1 = layer.FullConnLayer(input_dim=self.seq_dim,
                                         output_dim=self.hidden_dim,
                                         activation_function=act,
                                         initial_weights='AUTO',
                                         initial_bias=0.0,
                                         initial_offset=np.mean(sequence, axis=0).reshape(1, sequence.shape[1]),
                                         connections=None,
                                         dtype=np.float64)
            layer2 = layer.FullConnLayer(input_dim=self.hidden_dim,
                                         output_dim=self.seq_dim,
                                         activation_function=act,
                                         initial_weights='AUTO',
                                         initial_bias=0.0,
                                         initial_offset=0.,
                                         connections=None,
                                         dtype=np.float64)
            self.experiments = experiments.Model([layer1, layer2])
        self.trainer = trainer.GDTrainer(self.experiments)

        # Current pattern, used to return the next pattern by get_next_pattern
        # i.e. It represents the intrinsic state of the network
        self.current_state = None

        # If True the training sequence is not deleted after training, just for the case you want to keep it e.g. save
        # all to gard disk and want to recover the exact sequence later
        self.store_training_sequence = store_training_sequence

        # Store and if desired binarize training sequence
        self._training_sequence = sequence

    def train(self,
              epsilon=0.1,
              disturbance=0.05,
              max_epochs=1000,
              batch_size=1):
        """
        :param epsilon: Learning rate.
        :type epsilon: float

        :param disturbance: Percentage of noise added to the input.
        :type disturbance: float

        :param max_epochs: Maximum number of epochs for training.
        :type max_epochs: int

        :param batch_size: Batch size used during training.
        :type batch_size: int

        """
        # Roll sequence to get target output for the network.
        sequence_next = np.roll(self._training_sequence, -1, 0)

        for epoch in range(1, max_epochs + 1):
            for b in range(0, self._training_sequence.shape[0], batch_size):
                if self.experiments.num_layers == 1:
                    self.trainer.train(data=self._training_sequence[b:b + batch_size:],
                                       labels=[sequence_next[b:b + batch_size, :]],
                                       costs=[self.cost],
                                       reg_costs=[1],
                                       momentum=[0],
                                       epsilon=[epsilon],
                                       update_offsets=[0],
                                       corruptor=[RandomPermutation(disturbance), None],
                                       reg_L1Norm=[0],
                                       reg_L2Norm=[0],
                                       reg_sparseness=[0],
                                       desired_sparseness=[0],
                                       costs_sparseness=[None],
                                       restrict_gradient=0.0,
                                       restriction_norm='Mat')
                else:
                    self.trainer.train(data=self._training_sequence[b:b + batch_size:],
                                       labels=[None, sequence_next[b:b + batch_size, :]],
                                       costs=[None, self.cost],
                                       reg_costs=[0, 1],
                                       momentum=[0, 0],
                                       epsilon=[epsilon, epsilon],
                                       update_offsets=[0, 0],
                                       corruptor=[RandomPermutation(disturbance), None, None],
                                       reg_L1Norm=[0, 0],
                                       reg_L2Norm=[0, 0],
                                       reg_sparseness=[0, 0],
                                       desired_sparseness=[0, 0],
                                       costs_sparseness=[None, None],
                                       restrict_gradient=0.0,
                                       restriction_norm='Mat')

        self.current_state = np.atleast_2d(self._training_sequence[0])

        # If desired store original sequence
        if not self.store_training_sequence:
            self._training_sequence = None

    def calculate_full_loop_reconstruction(self, dataset, repeats=1, binarize_output=False):
        """ Calculated the full-loop reconstruction.
            Full loop means starting from a pattern it predicts the next, this is taken to predict the next, ... until
            you reach the prediction for the pattern we started from. This is done in parallel for all patterns in the
            given batch.

        :param dataset: Data samples
        :type dataset: numpy array

        :param repeats: Number of loops through the data
        :type repeats: int

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: Reconstruction
        :rtype: numpy array

        """
        # Calculate as full loop reconstruction for all given patterns
        result = None
        next_pattern = dataset
        for _ in range(repeats):
            for i in range(self.sequence_length):
                next_pattern = self.calculate_output(next_pattern, binarize_output)
                if result is None:
                    result = next_pattern
                else:
                    result = np.vstack((result, next_pattern))
        return result

    def set_current_state(self, offset_pattern, binarize_output=False):
        """ Sets/Overwrites the current pattern/state of the network.

        :param offset_pattern: new pattern/state.
        :type offset_pattern: numpy array

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        """
        self.current_state = offset_pattern
        if binarize_output:
            self.current_state = np.float64(self.current_state > 0.5)

    def get_current_state(self, binarize_output=False):
        """ Returns the current pattern/state of the network.

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: current pattern/state.
        :rtype: numpy array

        """
        if binarize_output:
            return np.float64(self.current_state > 0.5)
        return self.current_state

    def calculate_output(self, pattern, binarize_output=False):
        """ Returns the output for the network for a given pattern.
            The current pattern is NOT updated!!! see get_next_state to do so.

        :param pattern: Input pattern.
        :type pattern: numpy array

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: Output/prediction.
        :rtype: numpy array

        """
        output = self.experiments.forward_propagate(pattern)
        if binarize_output:
            output = np.float64(output > 0.5)
        return output

    def get_next_state(self, binarize_output=False):
        """ Returns and UPDATES the current pattern/state.
            i.e. Current is set to the next pattern.

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: next pattern/state.
        :rtype: numpy array

        """
        self.current_state = self.calculate_output(self.current_state, binarize_output)
        return self.get_current_state(binarize_output)
'''


class RandomNetwork(object):
    """ Random single layer network.

    """

    def __init__(self,
                 input_dim,
                 input_mean,
                 output_dim,
                 act=HebbSigmoid,
                 connection_matrix=None
                 ):
        """ Constructor for Random network.

        :param input_dim: Input dimensionality.
        :type input_dim: int.

        :param input_mean: Average input activity of the input.
        :type input_mean: float.

        :param output_dim: Output dimensionality.
        :type output_dim: int.

        :param act: Activation function.
        :type act: pydeep.base.activationfunction.

        :param connection_matrix: Connectivity matrix.
        :type input_dim: 2D numpy array.

        """
        # Store provided parameters
        self.input_dim = input_dim
        self.input_mean = input_mean
        self.output_dim = output_dim
        self.act = act
        self.model = model.Model([layer.FullConnLayer(input_dim=input_dim,
                                                      output_dim=output_dim,
                                                      activation_function=self.act,
                                                      initial_weights='AUTO',
                                                      initial_bias=0.0,
                                                      initial_offset=input_mean,
                                                      connections=connection_matrix,
                                                      dtype=np.float64)])

    def calculate_output(self, input_data, binarize_output=False):
        """ Calculates the output of the network.

        :param input_data: Input data.
        :type input_data: numpy array.

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :return: Output of the network.
        :rtype: numpy array

        """
        if binarize_output:
            return np.float64(self.model.forward_propagate(input_data) > 0.5)
        else:
            return self.model.forward_propagate(input_data)
