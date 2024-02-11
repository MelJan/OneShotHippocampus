""" Hippocampus models.

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
from hebbActivationFunction import HebbSigmoid
from pydeep.base.costfunction import SquaredError
from pydeep.base.corruptor import RandomPermutation,Dropout
import hippoLayer as Layer
from dataProvider import *
from os.path import isfile
import pydeep.misc.io as io


class HippoECCA3EC(object):
    """ Hippocampus experiments without Dentate gyrus and CA1.
    """

    def __init__(self,
                 EC_dim,
                 EC_activity,
                 CA3_dim,
                 CA3_activity,
                 CA3_capacity,
                 CA3_completion_loops,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 binarize_output=False,
                 load=False
                 ):
        """ Constructor.

        :param EC_dim: Dimensions in EC.
        :type EC_dim: int

        :param EC_activity: Average activity of EC data.
        :type EC_activity: numpy array (1,EC_dim) or scalar

        :param CA3_dim: Dimensions in CA3.
        :type EC_activity: int

        :param CA3_activity: Average activity of EC data.
        :type CA3_activity: numpy array (1,CA3_dim) or scalar

        :param CA3_capacity: Number of intrinsic states or sequence length.
        :type CA3_capacity: int

        :param CA3_completion_loops: Number of pattern completion loops in CA3.
        :type CA3_completion_loops: int

        :param act: Activation function, if single layer + Squared error + HebbSigmoid is chosen it performs
                    generalized Hebb learning.
        :type act: numpy array

        :param cost: Cost function, if single layer + Squared error + HebbSigmoid is chosen it performs generalized
                     Hebb learning.
        :type cost: pydeep.base.costfunction

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :param load: If True and hippocampus have been trained before before and stored to HD, hippocampus are loaded.
        :type load: bool
        """
        self.EC_dim = EC_dim
        self.EC_activity = EC_activity
        self.CA3_dim = CA3_dim
        self.CA3_capacity = CA3_capacity
        self.CA3_activity = CA3_activity
        self.CA3_completion_loops = CA3_completion_loops

        CA3_epsilon = 1.0
        CA3_disturbance = 0.1
        CA3_epochs = 100
        CA3_batchsize = 10
        file_name = "CA3_CA3_"+str(CA3_dim)+"_"+str(CA3_capacity)+"_"+str(CA3_activity)+"_"+str(CA3_completion_loops)+"_"+\
                    str(CA3_epsilon)+"_"+str(CA3_disturbance)+"_"+str(CA3_epochs)+"_"+str(CA3_batchsize)
        if isfile(file_name) and load:
            print("Loading CA3 network: "+file_name)
            self.CA3_CA3 = io.load_object(file_name,True,False)
        else:
            intrinsic_sequence = generate_binary_random_sequence(self.CA3_capacity, self.CA3_dim,
                                                                 self.CA3_dim * self.CA3_activity)
            self.CA3_CA3 = Layer.PredictiveAssociator(intrinsic_sequence, completion_loops=CA3_completion_loops)
            self.CA3_CA3.train(CA3_epsilon, CA3_disturbance, CA3_epochs, CA3_batchsize)
            io.save_object(self.CA3_CA3,file_name,True,False)

        self.EC_CA3 = Layer.HeteroAssociator(input_dim=self.EC_dim,
                                             input_mean=self.EC_activity,
                                             output_dim=self.CA3_dim,
                                             act=act,
                                             connection_matrix=None)#connection_matrix=np.random.randint(0,2,(self.EC_dim,CA3_dim)))

        self.CA3_EC = Layer.HeteroAssociator(input_dim=self.CA3_dim,
                                             input_mean=np.mean(self.CA3_CA3._training_sequence, axis=0).reshape(1, self.CA3_dim),
                                             output_dim=self.EC_dim,
                                             act=act,
                                             connection_matrix=None)#connection_matrix=np.random.randint(0,2,(CA3_dim,self.EC_dim)))
        self.binarize_output = binarize_output

    def store_data_point(self,
                         data_point,
                         epsilon_enc=0.1,
                         epsilon_dec=0.1,
                         disturbance=0.0,
                         iterations=1,
                         l1norm=0.0,
                         l2norm=0.0
                         ):
        """ Stores a new datapoint in the network.

        :param data_point: New datapoint to store.
        :type data_point: numpy array.

        :param epsilon_enc: Learning rate for encoder.
        :type epsilon_enc: float

        :param epsilon_dec: Learning rate for decoder.
        :type epsilon_dec: float

        :param disturbance: Noise factor (0,1) (percentage of neurons change activities).
        :type disturbance: float

        :param iterations: Number of updates performed.
        :type iterations: int

        :param l1norm: L1 weight decay term (0,1)
        :type l1norm: float

        :param l2norm: L2 eeight decay term (0,1)
        :type l2norm: float
        """
        # batch case
        if data_point.shape == (self.CA3_CA3.sequence_length,self.EC_dim):
            data = data_point
            intrinsic = self.CA3_CA3._training_sequence
            print "WARNING: BATCH LEARNING USED!!!!"
        else:
            data = data_point.reshape(1, self.EC_dim)
            intrinsic = self.CA3_CA3.get_current_state(binarize_output=self.binarize_output).reshape(1, self.CA3_dim)
        corr = Dropout(disturbance)
        if disturbance <= 0.0:
            corr = None
        self.EC_CA3.store_datapoint(input_data=data,
                                    output_data=intrinsic,
                                    epochs=iterations,
                                    epsilon=epsilon_enc,
                                    update_offsets=0.0,
                                    momentum=0.0,
                                    l1norm=l1norm,
                                    l2norm=l2norm,
                                    corruptor=corr)
        self.CA3_EC.store_datapoint(input_data=intrinsic,
                                    output_data=data,
                                    epochs=iterations,
                                    epsilon=epsilon_dec,
                                    update_offsets=0.0,
                                    momentum=0.0,
                                    l1norm=l1norm,
                                    l2norm=l2norm,
                                    corruptor=corr)
        self.CA3_CA3.get_next_state(binarize_output=self.binarize_output)

    def stabalize_memory(self,
                         epochs = 100,
                         epsilon = 0.1,
                         batch_size = 1
                        ):
        """ Recalls data from decoder and updates encoder.

        :param epochs: Number rof epochs recall cycles.
        :type epochs: float

        :param epsilon: Learning rate.
        :type epsilon: float

        """
        # batch case
        intrinsic = numx.copy(self.CA3_CA3._training_sequence)
        data = numx.copy(self.decode(self.CA3_CA3._training_sequence))

        for e in range(epochs):
            print e, batch_size,epsilon
            idx = numx.arange(data.shape[0])
            numx.random.shuffle(idx)
            data = data[idx]
            intrinsic = intrinsic[idx]
            for b in range(0, data.shape[0], batch_size):
                self.EC_CA3.store_datapoint(input_data=data[b:b + batch_size:],
                                           output_data=intrinsic[b:b + batch_size:],
                                           epochs=1,
                                           epsilon=epsilon,
                                           update_offsets=0.0,
                                           momentum=0.0,
                                           l1norm=0,
                                           l2norm=0,
                                           corruptor=None)
        '''
        intrinsic_sequence = self.encode(data)
        self.CA3_CA3 = Layer.PredictiveAssociator(intrinsic_sequence)
        self.CA3_CA3.train()

        self.CA3_EC.experiments.layers[0].weights = numx.random.randn(self.CA3_EC.experiments.layers[0].weights.shape[0],
                                                                self.CA3_EC.experiments.layers[0].weights.shape[1])*0.01
        self.CA3_EC.experiments.layers[0].bias *= 0

        for e in range(epochs):
            #for i in range(data.shape[0]):
            self.CA3_EC.store_datapoint(input_data=intrinsic,
                                            output_data=data,
                                            epochs=1,
                                            epsilon=epsilon,
                                            update_offsets=0.0,
                                            momentum=0.0,
                                            l1norm=0,
                                            l2norm=0,
                                            corruptor=None)
        '''


    def encode(self, data):
        """ Encodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.EC_CA3.calculate_output(data, binarize_output=self.binarize_output)

    def decode(self, data):
        """ Decodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.CA3_EC.calculate_output(data, binarize_output=self.binarize_output)

    def encode_decode(self, data):
        """ Encodes and decodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.decode(self.encode(data))

    def encode_next_decode(self, data):
        """ Encodes the data, gets the next intrinsic state and decodes it such that based on x_t , x_t+1 is calculated.

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        output = self.encode(data)
        output = self.CA3_CA3.calculate_output(output, binarize_output=self.binarize_output)
        return self.decode(output)

    def encode_previous_decode(self, data):
        """ Encodes the data, gets the previous intrinsic state and decodes it such that based on x_t , x_t-1 is calculated.

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        output = self.encode(data)
        output = self.CA3_CA3.calculate_input(output, binarize_output=self.binarize_output)
        return self.decode(output)

    def full_loop_encode_next_decode(self, data, forward=True):
        """ Calculates the full loop output. Each given data point is fed into the network, which predicts/outputs the
            next data point, which is fed to the network again. This is repeated for the entire sequence length.

        :param data: Input data, for each data point the full loop is done.
        :type data: numpy array

        :param data: Retrieval direction.
        :type data: bool

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        curr = data
        result = None
        for _ in range(self.CA3_capacity):
            if forward:
                curr = self.encode_next_decode(curr)
            else:
                curr = self.encode_previous_decode(curr)
            if result is None:
                result = curr
            else:
                result = np.vstack((result, curr))
        return result

    def full_intrinsic_loop_encode_next_decode(self, data, forward=True):
        """ Calculates the full intrinsic loop output. Each given data point is fed into, which triggers a state in CA3.
            The intrinsic dynamics are then used to reconstruct the full sequence.

        :param data: Input data, for each data point the full loop is done.
        :type data: numpy array

        :param data: Retrieval direction.
        :type data: bool

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        result = None
        curr = self.encode(data)
        for _ in range(self.CA3_capacity):
            if forward:
                curr = self.CA3_CA3.calculate_output(curr, binarize_output=self.binarize_output)
            else:
                curr = self.CA3_CA3.calculate_input(curr, binarize_output=self.binarize_output)
            output = self.decode(curr)
            if result is None:
                result = output
            else:
                result = np.vstack((result, output))
        return result

    def full_intrinsic_loop_encode_next_decode_return_last(self, data, forward=True):
        """ Calculates the full intrinsic loop output. Each given data point is fed into, which triggers a state in CA3.
            The intrinsic dynamics are then used to reconstruct the full sequence.

        :param data: Input data, for each data point the full loop is done.
        :type data: numpy array

        :param data: Retrieval direction.
        :type data: bool

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        result = None
        curr = self.encode(data)
        for _ in range(self.CA3_capacity):
            if forward:
                curr = self.CA3_CA3.calculate_output(curr, binarize_output=self.binarize_output)
            else:
                curr = self.CA3_CA3.calculate_input(curr, binarize_output=self.binarize_output)
        return self.decode(curr)

class HippoECDGCA3EC(HippoECCA3EC):
    """ Hippocampus experiments with Dentate gyrus.

    """

    def __init__(self,
                 EC_dim,
                 EC_activity,
                 DG_dim,
                 DG_activity,
                 CA3_dim,
                 CA3_activity,
                 CA3_capacity,
                 CA3_completion_loops,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 binarize_output=False,
                 load=False
                 ):
        """ Constructor.

        :param EC_dim: Dimensions in EC.
        :type EC_dim: int

        :param EC_activity: Average activity of EC data.
        :type EC_activity: numpy array (1,EC_dim) or scalar

        :param DG_dim: Dimensions in DG.
        :type DG_dim: int

        :param DG_activity: Average activity of DG.
        :type DG_activity: numpy array (1,DG_dim) or scalar

        :param CA3_dim: Dimensions in CA3.
        :type EC_activity: int

        :param CA3_activity: Average activity of EC data.
        :type CA3_activity: numpy array (1,CA3_dim) or scalar

        :param CA3_capacity: Number of intrinsic states or sequence length.
        :type CA3_capacity: int

        :param CA3_completion_loops: Number of pattern completion loops in CA3.
        :type CA3_completion_loops: int

        :param act: Activation function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type act: numpy array

        :param cost: Cost function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type cost: pydeep.base.costfunction

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :param load: If True and hippocampus have been trained before before and stored to HD, hippocampus are loaded.
        :type load: bool
        """
        self.EC_dim = EC_dim
        self.EC_activity = EC_activity
        self.DG_dim = DG_dim
        self.DG_activity = DG_activity
        self.CA3_dim = CA3_dim
        self.CA3_capacity = CA3_capacity
        self.CA3_activity = CA3_activity
        self.CA3_completion_loops = CA3_completion_loops

        CA3_epsilon = 1.0
        CA3_disturbance = 0.1
        CA3_epochs = 100
        CA3_batchsize = 10
        file_name = "CA3_CA3_" + str(CA3_dim) + "_" + str(CA3_capacity) + "_" + str(CA3_activity) + "_" + str(
            CA3_completion_loops) + "_" + \
                    str(CA3_epsilon) + "_" + str(CA3_disturbance) + "_" + str(CA3_epochs) + "_" + str(CA3_batchsize)
        print file_name
        if isfile(file_name) and load:
            print("Loading CA3 network: "+file_name)
            self.CA3_CA3 = io.load_object(file_name,True,False)
        else:
            intrinsic_sequence = generate_binary_random_sequence(self.CA3_capacity, self.CA3_dim,
                                                                 self.CA3_dim * self.CA3_activity)
            self.CA3_CA3 = Layer.PredictiveAssociator(intrinsic_sequence, completion_loops=CA3_completion_loops)
            self.CA3_CA3.train(CA3_epsilon, CA3_disturbance, CA3_epochs, CA3_batchsize)
            io.save_object(self.CA3_CA3, file_name,True,False)

        DG_epsilon = 100.0
        DG_disturbance = 0.0
        DG_epochs = 400
        file_name = "EC_DG_" + str(EC_dim) + "_"+ str(DG_dim) + "_" + str(EC_activity)+ "_" + str(DG_activity) + "_" + \
                    str(DG_epsilon) + "_" + str(DG_disturbance) + "_" + str(DG_epochs)
        print file_name
        if isfile(file_name) and load:
            print("Loading DG network: "+file_name)
            self.EC_DG = io.load_object(file_name,True,False)
        else:
            self.EC_DG = Layer.AutoAssociator(input_dim=EC_dim,
                                              input_mean=EC_activity,
                                              output_dim=DG_dim,
                                              output_mean=DG_activity,
                                              act=act,
                                              connection_matrix=None)
            self.EC_DG.train(epochs=DG_epochs,
                             epsilon=DG_epsilon,
                             update_offsets=0.0,
                             momentum=0.0,
                             l1norm=0.0,
                             l2norm=0.0,
                             corruptor=[Dropout(DG_disturbance), None, None])
            io.save_object(self.EC_DG, file_name,True,False)

        self.DG_CA3 = Layer.HeteroAssociator(input_dim=self.DG_dim,
                                             input_mean=self.DG_activity,
                                             output_dim=self.CA3_dim,
                                             act=act,#hebbActivationFunction.HebbKMax(self.CA3_dim * self.CA3_activity),
                                             connection_matrix=None)

        self.CA3_EC = Layer.HeteroAssociator(input_dim=self.CA3_dim,
                                             input_mean=np.mean(self.CA3_CA3._training_sequence,
                                                                axis=0).reshape(1, self.CA3_dim),
                                             output_dim=self.EC_dim,
                                             act=act,
                                             connection_matrix=None)
        self.binarize_output = binarize_output

    def store_data_point(self,
                         data_point,
                         epsilon_enc=0.1,
                         epsilon_dec=0.1,
                         disturbance=0.0,
                         iterations=10,
                         l1norm=0.0,
                         l2norm=0.0
                         ):
        """ Stores a new datapoint in the network.

        :param data_point: New datapoint to store.
        :type data_point: numpy array.

        :param epsilon_enc: Learning rate for encoder.
        :type epsilon_enc: float

        :param epsilon_dec: Learning rate for decoder.
        :type epsilon_dec: float

        :param disturbance: Noise factor (0,1) (percentage of neurons change activities).
        :type disturbance: float

        :param iterations: Number of updates performed.
        :type iterations: int

        :param l1norm: L1 weight decay term (0,1)
        :type l1norm: float

        :param l2norm: L2 eeight decay term (0,1)
        :type l2norm: float
        """
        # batch case
        if data_point.shape == (self.CA3_CA3.sequence_length,self.EC_dim):
            data_EC = data_point
            data_DG = self.EC_DG.calculate_output(data_EC, binarize_output=self.binarize_output)
            intrinsic = self.CA3_CA3._training_sequence
            print "WARNING: BATCH LEARNING USED!!!!"
        else:
            data_EC = data_point.reshape(1, self.EC_dim)
            data_DG = self.EC_DG.calculate_output(data_EC, binarize_output=self.binarize_output)
            intrinsic = self.CA3_CA3.get_current_state(binarize_output=self.binarize_output).reshape(1, self.CA3_dim)

        corr = Dropout(disturbance)
        if disturbance <= 0.0:
            corr = None
        self.DG_CA3.store_datapoint(input_data=data_DG,
                                    output_data=intrinsic,
                                    epochs=iterations,
                                    epsilon=epsilon_enc,
                                    update_offsets=0.0,
                                    momentum=0.0,
                                    l1norm=l1norm,
                                    l2norm=l2norm,
                                    corruptor=corr)
        self.CA3_EC.store_datapoint(input_data=intrinsic,
                                    output_data=data_EC,
                                    epochs=iterations,
                                    epsilon=epsilon_dec,
                                    update_offsets=0.0,
                                    momentum=0.0,
                                    l1norm=l1norm,
                                    l2norm=l2norm,
                                    corruptor=corr)
        self.CA3_CA3.get_next_state(binarize_output=self.binarize_output)

    def stabalize_memory(self,
                         epochs = 100,
                         epsilon = 0.1,
                         batch_size = 1
                        ):
        """ Recalls data from decoder and updates encoder.

        :param epochs: Number rof epochs recall cycles.
        :type epochs: float

        :param epsilon: Learning rate.
        :type epsilon: float

        """
        # batch case
        intrinsic = numx.copy(self.CA3_CA3._training_sequence)
        data = numx.copy(self.EC_DG.calculate_output(self.decode(self.CA3_CA3._training_sequence), binarize_output=self.binarize_output))

        for e in range(epochs):
            idx = numx.arange(data.shape[0])
            numx.random.shuffle(idx)
            data = data[idx]
            intrinsic = intrinsic[idx]
            for b in range(0, data.shape[0], batch_size):
                self.DG_CA3.store_datapoint(input_data=data[b:b + batch_size:],
                                           output_data=intrinsic[b:b + batch_size:],
                                           epochs=1,
                                           epsilon=epsilon,
                                           update_offsets=0.0,
                                           momentum=0.0,
                                           l1norm=0,
                                           l2norm=0,
                                           corruptor=None)

    def encode(self, data):
        """ Encodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.DG_CA3.calculate_output(self.EC_DG.calculate_output(data, binarize_output=self.binarize_output),
                                            binarize_output=self.binarize_output)

    def decode(self, data):
        """ Decodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.CA3_EC.calculate_output(data, binarize_output=self.binarize_output)

class HippoECDGCA3CA1EC(HippoECCA3EC):
    """ Hippocampus experiments with Dentate gyrus and CA1.

    """

    def __init__(self,
                 EC_dim,
                 EC_activity,
                 DG_dim,
                 DG_activity,
                 CA3_dim,
                 CA3_activity,
                 CA3_capacity,
                 CA1_dim,
                 CA1_activity,
                 CA3_completion_loops,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 binarize_output=False,
                 load=False
                 ):
        """ Constructor.

        :param EC_dim: Dimensions in EC.
        :type EC_dim: int

        :param EC_activity: Average activity of EC data.
        :type EC_activity: numpy array (1,EC_dim) or scalar

        :param DG_dim: Dimensions in DG.
        :type DG_dim: int

        :param DG_activity: Average activity of DG.
        :type DG_activity: numpy array (1,DG_dim) or scalar

        :param CA3_dim: Dimensions in CA3.
        :type EC_activity: int

        :param CA3_activity: Average activity of EC data.
        :type CA3_activity: numpy array (1,CA3_dim) or scalar

        :param CA3_capacity: Number of intrinsic states or sequence length.
        :type CA3_capacity: int

        :param CA1_dim: Dimensions in CA1.
        :type CA1_dim: int

        :param CA1_activity: Average activity of CA1.
        :type CA1_activity: numpy array (1,CA1_dim) or scalar

        :param CA3_completion_loops: Number of pattern completion loops in CA3.
        :type CA3_completion_loops: int

        :param act: Activation function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type act: numpy array

        :param cost: Cost function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type cost: pydeep.base.costfunction

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :param load: If True and hippocampus have been trained before before and stored to HD, hippocampus are loaded.
        :type load: bool
        """
        self.EC_dim = EC_dim
        self.EC_activity = EC_activity
        self.DG_dim = DG_dim
        self.DG_activity = DG_activity
        self.CA3_dim = CA3_dim
        self.CA3_capacity = CA3_capacity
        self.CA3_activity = CA3_activity
        self.CA1_dim = CA1_dim
        self.CA1_activity = CA1_activity
        self.CA3_completion_loops = CA3_completion_loops

        CA3_epsilon = 1.0
        CA3_disturbance = 0.1
        CA3_epochs = 100
        CA3_batchsize = 10
        file_name = "CA3_CA3_" + str(CA3_dim) + "_" + str(CA3_capacity) + "_" + str(CA3_activity) + "_" + str(
            CA3_completion_loops) + "_" + \
                    str(CA3_epsilon) + "_" + str(CA3_disturbance) + "_" + str(CA3_epochs) + "_" + str(CA3_batchsize)
        if isfile(file_name) and load:
            self.CA3_CA3 = io.load_object(file_name,True,False)
        else:
            intrinsic_sequence = generate_binary_random_sequence(self.CA3_capacity, self.CA3_dim,
                                                                 self.CA3_dim * self.CA3_activity)
            self.CA3_CA3 = Layer.PredictiveAssociator(intrinsic_sequence, completion_loops=CA3_completion_loops)
            self.CA3_CA3.train(CA3_epsilon, CA3_disturbance, CA3_epochs, CA3_batchsize)
            io.save_object(self.CA3_CA3, file_name,True,False)

        DG_epsilon = 100.0
        DG_disturbance = 0.0
        DG_epochs = 400
        file_name = "EC_DG_" + str(EC_dim) + "_"+ str(DG_dim) + "_" + str(EC_activity)+ "_" + str(DG_activity) + "_" + \
                    str(DG_epsilon) + "_" + str(DG_disturbance) + "_" + str(DG_epochs)
        if isfile(file_name) and load:
            self.EC_DG = io.load_object(file_name,True,False)
        else:
            self.EC_DG = Layer.AutoAssociator(input_dim=EC_dim,
                                              input_mean=EC_activity,
                                              output_dim=DG_dim,
                                              output_mean=DG_activity,
                                              act=act,
                                              connection_matrix=None)
            self.EC_DG.train(epochs=DG_epochs,
                             epsilon=DG_epsilon,
                             update_offsets=0.0,
                             momentum=0.0,
                             l1norm=0.0,
                             l2norm=0.0,
                             corruptor=[Dropout(DG_disturbance), None, None])
            io.save_object(self.EC_DG, file_name,True,False)

        CA1_epsilon = 0.2
        CA1_disturbance = 0.0
        CA1_epochs = 1000
        file_name = "CA1_EC_" + str(CA1_dim) + "_"+ str(EC_dim) + "_" + str(CA1_activity)+ "_" + str(EC_activity) + "_" + \
                    str(CA1_epsilon) + "_" + str(CA1_disturbance) + "_" + str(CA1_epochs)
        if isfile(file_name) and load:
            self.CA1_EC = io.load_object(file_name,True,False)
        else:
            self.CA1_EC = Layer.AutoAssociator(input_dim=EC_dim,
                                               input_mean=EC_activity,
                                               output_dim=CA1_dim,
                                               output_mean=CA1_activity,
                                               act=act,
                                               connection_matrix=None)
            self.CA1_EC.train(epochs=CA1_epochs,
                              epsilon=CA1_epsilon,
                              update_offsets=0.0,
                              momentum=0.0,
                              l1norm=0.0,
                              l2norm=0.0,
                              corruptor=[None,Dropout(CA1_disturbance), None])
            io.save_object(self.CA1_EC, file_name,True,False)

        self.DG_CA3 = Layer.HeteroAssociator(input_dim=self.DG_dim,
                                             input_mean=self.DG_activity,
                                             output_dim=self.CA3_dim,
                                             act=act,
                                             connection_matrix=None)

        self.CA3_CA1 = Layer.HeteroAssociator(input_dim=self.CA3_dim,
                                              input_mean=np.mean(self.CA3_CA3._training_sequence,
                                                                 axis=0).reshape(1, self.CA3_dim),
                                              output_dim=self.CA1_dim,
                                              act=act,
                                              connection_matrix=None)

        self.binarize_output = binarize_output

    def store_data_point(self,
                         data_point,
                         epsilon_enc=0.1,
                         epsilon_dec=0.1,
                         disturbance=0.0,
                         iterations=10,
                         l1norm=0.0,
                         l2norm=0.0
                         ):
        """ Stores a new datapoint in the network.

        :param data_point: New datapoint to store.
        :type data_point: numpy array.

        :param epsilon_enc: Learning rate for encoder.
        :type epsilon_enc: float

        :param epsilon_dec: Learning rate for decoder.
        :type epsilon_dec: float

        :param disturbance: Noise factor (0,1) (percentage of neurons change activities).
        :type disturbance: float

        :param iterations: Number of updates performed.
        :type iterations: int

        :param l1norm: L1 weight decay term (0,1)
        :type l1norm: float

        :param l2norm: L2 weight decay term (0,1)
        :type l2norm: float
        """
        data_EC = data_point.reshape(1, self.EC_dim)
        data_DG = self.EC_DG.calculate_output(data_EC, binarize_output=self.binarize_output)
        data_CA1 = self.CA1_EC.calculate_output(data_EC, binarize_output=self.binarize_output)
        intrinsic = self.CA3_CA3.get_current_state(binarize_output=self.binarize_output).reshape(1, self.CA3_dim)
        corr = Dropout(disturbance)
        if disturbance <= 0.0:
            corr = None
        self.DG_CA3.store_datapoint(input_data=data_DG,
                                    output_data=intrinsic,
                                    epochs=iterations,
                                    epsilon=epsilon_enc,
                                    update_offsets=0.0,
                                    momentum=0.0,
                                    l1norm=l1norm,
                                    l2norm=l2norm,
                                    corruptor=corr)
        self.CA3_CA1.store_datapoint(input_data=intrinsic,
                                     output_data=data_CA1,
                                     epochs=iterations,
                                     epsilon=epsilon_dec,
                                     update_offsets=0.0,
                                     momentum=0.0,
                                     l1norm=l1norm,
                                     l2norm=l2norm,
                                     corruptor=corr)
        self.CA3_CA3.get_next_state(binarize_output=self.binarize_output)

    def stabalize_memory(self,
                         epochs = 100,
                         epsilon = 0.1,
                         batch_size = 1
                        ):
        """ Recalls data from decoder and updates encoder.

        :param epochs: Number rof epochs recall cycles.
        :type epochs: float

        :param epsilon: Learning rate.
        :type epsilon: float

        """
        # batch case
        data = numx.copy(self.EC_DG.calculate_output(self.decode(self.CA3_CA3._training_sequence), binarize_output=self.binarize_output))
        intrinsic = numx.copy(self.CA3_CA3._training_sequence)

        for e in range(epochs):
            idx = numx.arange(data.shape[0])
            numx.random.shuffle(idx)
            data = data[idx]
            intrinsic = intrinsic[idx]
            for b in range(0, data.shape[0], batch_size):
                self.DG_CA3.store_datapoint(input_data=data[b:b + batch_size:],
                                            output_data=intrinsic[b:b + batch_size:],
                                            epochs=1,
                                            epsilon=epsilon,
                                            update_offsets=0.0,
                                            momentum=0.0,
                                            l1norm=0,
                                            l2norm=0,
                                            corruptor=None)

    def encode(self, data):
        """ Encodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.DG_CA3.calculate_output(self.EC_DG.calculate_output(data, binarize_output=self.binarize_output),
                                            binarize_output=self.binarize_output)

    def decode(self, data):
        """ Decodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.CA1_EC.calculate_input(self.CA3_CA1.calculate_output(data, binarize_output=self.binarize_output),
                                           binarize_output=self.binarize_output)


########################################################################################################################
########################################################################################################################
#################################################   EXPETRIMENTAL   ####################################################
########################################################################################################################
########################################################################################################################


'''
class HippoECDGCA3EC_2(HippoECCA3EC):
    """ Hippocampus experiments with Dentate gyrus.

    """

    def __init__(self,
                 EC_dim,
                 EC_activity,
                 DG_dim,
                 DG_activity,
                 CA3_dim,
                 CA3_activity,
                 CA3_capacity,
                 CA3_completion_loops,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 binarize_output=False,
                 load=False
                 ):
        """ Constructor.

        :param EC_dim: Dimensions in EC.
        :type EC_dim: int

        :param EC_activity: Average activity of EC data.
        :type EC_activity: numpy array (1,EC_dim) or scalar

        :param DG_dim: Dimensions in DG.
        :type DG_dim: int

        :param DG_activity: Average activity of DG.
        :type DG_activity: numpy array (1,DG_dim) or scalar

        :param CA3_dim: Dimensions in CA3.
        :type EC_activity: int

        :param CA3_activity: Average activity of EC data.
        :type CA3_activity: numpy array (1,CA3_dim) or scalar

        :param CA3_capacity: Number of intrinsic states or sequence length.
        :type CA3_capacity: int

        :param CA3_completion_loops: Number of pattern completion loops in CA3.
        :type CA3_completion_loops: int

        :param act: Activation function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type act: numpy array

        :param cost: Cost function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type cost: pydeep.base.costfunction

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool

        :param load: If True and hippocampus have been trained before before and stored to HD, hippocampus are loaded.
        :type load: bool
        """
        self.EC_dim = EC_dim
        self.EC_activity = EC_activity
        self.DG_dim = DG_dim
        self.DG_activity = DG_activity
        self.CA3_dim = CA3_dim
        self.CA3_capacity = CA3_capacity
        self.CA3_activity = CA3_activity
        self.CA3_completion_loops = CA3_completion_loops

        CA3_epsilon = 1.0
        CA3_disturbance = 0.0
        CA3_epochs = 1000
        CA3_batchsize = 10
        file_name = "CA3_CA3_" + str(CA3_dim) + "_" + str(CA3_capacity) + "_" + str(CA3_activity) + "_" + str(
            CA3_completion_loops) + "_" + \
                    str(CA3_epsilon) + "_" + str(CA3_disturbance) + "_" + str(CA3_epochs) + "_" + str(CA3_batchsize)
        print file_name
        if isfile(file_name) and load:
            self.CA3_CA3 = io.load_object(file_name)
        else:
            intrinsic_sequence_CA3 = generate_binary_random_sequence(self.CA3_capacity, self.CA3_dim,
                                                                 self.CA3_dim * self.CA3_activity)
            self.CA3_CA3 = Layer.PredictiveAssociator(intrinsic_sequence_CA3, completion_loops=CA3_completion_loops)
            self.CA3_CA3.train(CA3_epsilon, CA3_disturbance, CA3_epochs, CA3_batchsize)
            io.save_object(self.CA3_CA3, file_name)

        DG_epsilon = 0.1
        DG_disturbance = 0.0
        DG_epochs = 1000
        DG_batchsize = 10
        file_name = "DG_CA3_" + str(DG_dim) + "_"+ str(CA3_dim) + "_" + str(DG_activity)+ "_" + str(CA3_activity) + "_" + \
                    str(DG_epsilon) + "_" + str(DG_disturbance) + "_" + str(DG_epochs)

        self.intrinsic_sequence_DG = None
        self.current_index = 0
        print file_name
        if isfile(file_name) and load:
            self.DG_CA3 = io.load_object(file_name)
            self.intrinsic_sequence_DG = io.load_object(file_name+"_sequence")
        else:
            self.DG_CA3 = Layer.HeteroAssociator(input_dim=self.DG_dim,
                                                 input_mean=self.DG_activity,
                                                 output_dim=self.CA3_dim,
                                                 act=act,
                                                 connection_matrix=None)
            self.intrinsic_sequence_DG = generate_binary_random_sequence(self.CA3_capacity, self.DG_dim,
                                                                         self.DG_dim * self.DG_activity)
            for e in range(DG_epochs):
                for b in range(0, self.intrinsic_sequence_DG.shape[0], DG_batchsize):
                    self.DG_CA3.store_datapoint(input_data=self.intrinsic_sequence_DG[b:b + DG_batchsize:],
                                                output_data=self.CA3_CA3._training_sequence[b:b + DG_batchsize:],
                                                update_offsets=0.0,
                                                momentum=0.0,
                                                l1norm=0.0,
                                                l2norm=0.0,
                                                corruptor=Dropout(DG_disturbance))
            print numx.mean(numx.abs(self.DG_CA3.calculate_output(self.intrinsic_sequence_DG)-self.CA3_CA3._training_sequence))
            io.save_object(self.DG_CA3, file_name)
            io.save_object(self.intrinsic_sequence_DG, file_name+"_sequence")

        self.EC_DG = Layer.HeteroAssociator(input_dim=self.EC_dim,
                                                input_mean=self.EC_activity,
                                                output_dim=self.DG_dim,
                                                act=act,
                                                connection_matrix=None)
        self.CA3_EC = Layer.HeteroAssociator(input_dim=self.CA3_dim,
                                             input_mean=np.mean(self.CA3_CA3._training_sequence,
                                                                axis=0).reshape(1, self.CA3_dim),
                                             output_dim=self.EC_dim,
                                             act=act,
                                             connection_matrix=None)
        self.binarize_output = binarize_output

    def store_data_point(self,
                         data_point,
                         epsilon_enc=0.1,
                         epsilon_dec=0.1,
                         disturbance=0.0,
                         iterations=10,
                         l1norm=0.0,
                         l2norm=0.0
                         ):
        """ Stores a new datapoint in the network.

        :param data_point: New datapoint to store.
        :type data_point: numpy array.

        :param epsilon_enc: Learning rate for encoder.
        :type epsilon_enc: float

        :param epsilon_dec: Learning rate for decoder.
        :type epsilon_dec: float

        :param disturbance: Noise factor (0,1) (percentage of neurons change activities).
        :type disturbance: float

        :param iterations: Number of updates performed.
        :type iterations: int

        :param l1norm: L1 weight decay term (0,1)
        :type l1norm: float

        :param l2norm: L2 eeight decay term (0,1)
        :type l2norm: float
        """
        # batch case
        if data_point.shape == (self.CA3_CA3.sequence_length,self.EC_dim):
            data_EC = data_point
            data_DG = self.intrinsic_sequence_DG
            data_CA3 = self.CA3_CA3._training_sequence
            print "WARNING: BATCH LEARNING USED!!!!"
        else:
            data_EC = data_point.reshape(1, self.EC_dim)
            data_DG = self.intrinsic_sequence_DG[self.current_index % self.CA3_capacity].reshape(1, self.DG_dim)
            data_CA3 = self.CA3_CA3.get_current_state(binarize_output=self.binarize_output).reshape(1, self.CA3_dim)

        corr = Dropout(disturbance)
        if disturbance <= 0.0:
            corr = None
        self.EC_DG.store_datapoint(input_data=data_EC,
                                    output_data=data_DG,
                                    epochs=iterations,
                                    epsilon=epsilon_enc,
                                    update_offsets=0.0,
                                    momentum=0.0,
                                    l1norm=l1norm,
                                    l2norm=l2norm,
                                    corruptor=corr)
        self.CA3_EC.store_datapoint(input_data=data_CA3,
                                    output_data=data_EC,
                                    epochs=iterations,
                                    epsilon=epsilon_dec,
                                    update_offsets=0.0,
                                    momentum=0.0,
                                    l1norm=l1norm,
                                    l2norm=l2norm,
                                    corruptor=corr)
        self.current_index = (self.current_index +1) % self.CA3_capacity
        self.CA3_CA3.get_next_state(binarize_output=self.binarize_output)


    def stabalize_memory(self,
                         epochs = 100,
                         epsilon = 0.1
                        ):
        """ Recalls data from decoder and updates encoder.

        :param epochs: Number rof epochs recall cycles.
        :type epochs: float

        :param epsilon: Learning rate.
        :type epsilon: float

        """
        # batch case
        intrinsic = self.intrinsic_sequence_DG
        data = self.decode(self.CA3_CA3._training_sequence)

        for e in range(epochs):
            for i in range(data.shape[0]):
                self.EC_DG.store_datapoint(input_data=data[i],
                                           output_data=intrinsic[i],
                                           epochs=1,
                                           epsilon=epsilon,
                                           update_offsets=0.0,
                                           momentum=0.0,
                                           l1norm=0,
                                           l2norm=0,
                                           corruptor=None)

    def encode(self, data):
        """ Encodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.DG_CA3.calculate_output(self.EC_DG.calculate_output(data, binarize_output=self.binarize_output),
                                            binarize_output=self.binarize_output)

    def decode(self, data):
        """ Decodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.CA3_EC.calculate_output(data, binarize_output=self.binarize_output)
'''

'''

class HippoECDGCA3CA1EC2(HippoECCA3EC):
    """ Hippocampus experiments with Dentate gyrus and CA1.

    """

    def __init__(self,
                 EC_dim,
                 EC_activity,
                 DG_dim,
                 DG_activity,
                 CA3_dim,
                 CA3_activity,
                 CA3_capacity,
                 CA1_dim,
                 CA1_activity,
                 CA3_completion_loops,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 binarize_output=False
                 ):
        """ Constructor.

        :param EC_dim: Dimensions in EC.
        :type EC_dim: int

        :param EC_activity: Average activity of EC data.
        :type EC_activity: numpy array (1,EC_dim) or scalar

        :param DG_dim: Dimensions in DG.
        :type DG_dim: int

        :param DG_activity: Average activity of DG.
        :type DG_activity: numpy array (1,DG_dim) or scalar

        :param CA3_dim: Dimensions in CA3.
        :type EC_activity: int

        :param CA3_activity: Average activity of EC data.
        :type CA3_activity: numpy array (1,CA3_dim) or scalar

        :param CA3_capacity: Number of intrinsic states or sequence length.
        :type CA3_capacity: int

        :param CA1_dim: Dimensions in CA1.
        :type CA1_dim: int

        :param CA1_activity: Average activity of CA1.
        :type CA1_activity: numpy array (1,CA1_dim) or scalar

        :param CA3_completion_loops: Number of pattern completion loops in CA3.
        :type CA3_completion_loops: int

        :param act: Activation function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type act: numpy array

        :param cost: Cost function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type cost: pydeep.base.costfunction

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool
        """
        self.EC_dim = EC_dim
        self.EC_activity = EC_activity
        self.DG_dim = DG_dim
        self.DG_activity = DG_activity
        self.CA3_dim = CA3_dim
        self.CA3_capacity = CA3_capacity
        self.CA3_activity = CA3_activity
        self.CA1_dim = CA1_dim
        self.CA1_activity = CA1_activity
        self.CA3_completion_loops = CA3_completion_loops

        CA3_epsilon = 1.0
        CA3_disturbance = 0.0
        CA3_epochs = 1000
        CA3_batchsize = 10
        file_name = "CA3_CA3_" + str(CA3_dim) + "_" + str(CA3_capacity) + "_" + str(CA3_activity) + "_" + str(
            CA3_completion_loops) + "_" + \
                    str(CA3_epsilon) + "_" + str(CA3_disturbance) + "_" + str(CA3_epochs) + "_" + str(CA3_batchsize)
        if isfile(file_name):
            self.CA3_CA3 = io.load_object(file_name)
        else:
            intrinsic_sequence = generate_binary_random_sequence(self.CA3_capacity, self.CA3_dim,
                                                                 self.CA3_dim * self.CA3_activity)
            self.CA3_CA3 = Layer.PredictiveAssociator(intrinsic_sequence, completion_loops=CA3_completion_loops)
            self.CA3_CA3.train(CA3_epsilon, CA3_disturbance, CA3_epochs, CA3_batchsize)
            io.save_object(self.CA3_CA3, file_name)

        DG_epsilon = 1.0
        DG_disturbance = 0.0
        DG_epochs = 1000
        file_name = "EC_DG_" + str(EC_dim) + "_" + str(DG_dim) + "_" + str(EC_activity) + "_" + str(
            DG_activity) + "_" + \
                    str(DG_epsilon) + "_" + str(DG_disturbance) + "_" + str(DG_epochs)
        if isfile(file_name):
            self.EC_DG = io.load_object(file_name)
        else:
            self.EC_DG = Layer.AutoAssociator(input_dim=EC_dim,
                                              input_mean=EC_activity,
                                              output_dim=DG_dim,
                                              output_mean=DG_activity,
                                              act=act,
                                              connection_matrix=None)
            self.EC_DG.train(epochs=DG_epochs,
                             epsilon=DG_epsilon,
                             update_offsets=0.0,
                             momentum=0.0,
                             l1norm=0.0,
                             l2norm=0.0,
                             corruptor=[Dropout(DG_disturbance), None, None])
            io.save_object(self.EC_DG, file_name)

        CA1_epsilon = 0.1
        CA1_disturbance = 0.0
        CA1_epochs = 1000
        file_name = "CA3_CA1_" + str(CA3_dim) + "_" + str(CA1_dim) + "_" + str(CA3_activity) + "_" + str(
            CA1_activity) + "_" + \
                    str(CA1_epsilon) + "_" + str(CA1_disturbance) + "_" + str(CA1_epochs)
        if isfile(file_name):
            self.CA3_CA1 = io.load_object(file_name)
        else:
            self.CA3_CA1 = Layer.AutoAssociator(input_dim=CA3_dim,
                                               input_mean=CA3_activity,
                                               output_dim=CA1_dim,
                                               output_mean=CA1_activity,
                                               act=act,
                                               connection_matrix=None)
            self.CA3_CA1.train(epochs=CA1_epochs,
                              epsilon=CA1_epsilon,
                              update_offsets=0.0,
                              momentum=0.0,
                              l1norm=0.0,
                              l2norm=0.0,
                              corruptor=[Dropout(CA1_disturbance), Dropout(CA1_disturbance), None])
            io.save_object(self.CA3_CA1, file_name)

        self.DG_CA3 = Layer.HeteroAssociator(input_dim=self.DG_dim,
                                             input_mean=self.DG_activity,
                                             output_dim=self.CA3_dim,
                                             act=act,
                                             connection_matrix=None)

        self.CA1_EC = Layer.HeteroAssociator(input_dim=self.CA1_dim,
                                             input_mean=self.CA1_activity,
                                             output_dim=self.EC_dim,
                                             act=act,
                                             connection_matrix=None)

        self.binarize_output = binarize_output

    def store_data_point(self,
                         data_point,
                         epsilon_enc=0.1,
                         epsilon_dec=0.1,
                         disturbance=0.0,
                         iterations=10,
                         l1norm=0.0,
                         l2norm=0.0
                         ):
        """ Stores a new datapoint in the network.

        :param data_point: New datapoint to store.
        :type data_point: numpy array.

        :param epsilon_enc: Learning rate for encoder.
        :type epsilon_enc: float

        :param epsilon_dec: Learning rate for decoder.
        :type epsilon_dec: float

        :param disturbance: Noise factor (0,1) (percentage of neurons change activities).
        :type disturbance: float

        :param iterations: Number of updates performed.
        :type iterations: int

        :param l1norm: L1 weight decay term (0,1)
        :type l1norm: float

        :param l2norm: L2 weight decay term (0,1)
        :type l2norm: float
        """
        data_EC = data_point.reshape(1, self.EC_dim)
        data_DG = self.EC_DG.calculate_output(data_EC, binarize_output=self.binarize_output)
        intrinsic = self.CA3_CA3.get_current_state(binarize_output=self.binarize_output).reshape(1, self.CA3_dim)
        data_CA1 = self.CA3_CA1.calculate_output(intrinsic, binarize_output=self.binarize_output)
        corr = Dropout(disturbance)
        if disturbance <= 0.0:
            corr = None
        self.DG_CA3.store_datapoint(input_data=data_DG,
                                    output_data=intrinsic,
                                    epochs=iterations,
                                    epsilon=epsilon_enc,
                                    update_offsets=0.0,
                                    momentum=0.0,
                                    l1norm=l1norm,
                                    l2norm=l2norm,
                                    corruptor=corr)
        self.CA1_EC.store_datapoint(input_data=data_CA1,
                                     output_data=data_EC,
                                     epochs=iterations,
                                     epsilon=epsilon_dec,
                                     update_offsets=0.0,
                                     momentum=0.0,
                                     l1norm=l1norm,
                                     l2norm=l2norm,
                                     corruptor=corr)
        self.CA3_CA3.get_next_state(binarize_output=self.binarize_output)

    def encode(self, data):
        """ Encodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.DG_CA3.calculate_output(self.EC_DG.calculate_output(data, binarize_output=self.binarize_output),
                                            binarize_output=self.binarize_output)

    def decode(self, data):
        """ Decodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.CA1_EC.calculate_output(
            self.CA3_CA1.calculate_output(data, binarize_output=self.binarize_output),
            binarize_output=self.binarize_output)

class Hippo2ECDGCA3CA1EC(HippoECCA3EC):
    """ Hippocampus experiments with Dentate gyrus and CA1.

    """

    def __init__(self,
                 EC_dim,
                 EC_activity,
                 DG_dim,
                 DG_activity,
                 CA3_dim,
                 CA3_activity,
                 CA3_capacity,
                 CA1_dim,
                 CA1_activity,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 binarize_output=False
                 ):
        """ Constructor.

        :param EC_dim: Dimensions in EC.
        :type EC_dim: int

        :param EC_activity: Average activity of EC data.
        :type EC_activity: numpy array (1,EC_dim) or scalar

        :param DG_dim: Dimensions in DG.
        :type DG_dim: int

        :param DG_activity: Average activity of DG.
        :type DG_activity: numpy array (1,DG_dim) or scalar

        :param CA3_dim: Dimensions in CA3.
        :type EC_activity: int

        :param CA3_activity: Average activity of EC data.
        :type CA3_activity: numpy array (1,CA3_dim) or scalar

        :param CA3_capacity: Number of intrinsic states or sequence length.
        :type CA3_capacity: int

        :param CA1_dim: Dimensions in CA1.
        :type CA1_dim: int

        :param CA1_activity: Average activity of CA1.
        :type CA1_activity: numpy array (1,CA1_dim) or scalar

        :param act: Activation function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type act: numpy array

        :param cost: Cost function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type cost: pydeep.base.costfunction

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool
        """
        self.EC_dim = EC_dim
        self.EC_activity = EC_activity
        self.DG_dim = DG_dim
        self.DG_activity = DG_activity
        self.CA3_dim = CA3_dim
        self.CA3_capacity = CA3_capacity
        self.CA3_activity = CA3_activity
        self.CA1_dim = CA1_dim
        self.CA1_activity = CA1_activity

        intrinsic_sequence = generate_binary_random_sequence(self.CA3_capacity, self.CA3_dim,
                                                             self.CA3_dim * self.CA3_activity)
        self.CA3_CA3 = Layer.PredictiveAssociator(intrinsic_sequence)
        self.CA3_CA3.train(0.1, 0.1, 1000, 10)

        self.EC_DG = Layer.AutoAssociator(input_dim=EC_dim,
                                          input_mean=EC_activity,
                                          output_dim=DG_dim,
                                          output_mean=DG_activity,
                                          act=act,
                                          sparse=True,
                                          connection_matrix=None)
        self.EC_DG.train(epochs=1000,
                         epsilon=0.1,
                         update_offsets=0.0,
                         momentum=0.0,
                         l1norm=0.0,
                         l2norm=0.0,
                         corruptor=[Dropout(0.1), None, None])

        self.CA3_CA1 = Layer.AutoAssociator(input_dim=CA3_dim,
                                            input_mean=np.mean(intrinsic_sequence, axis=0).reshape(1, self.CA3_dim),
                                            output_dim=CA1_dim,
                                            output_mean=CA1_activity,
                                            act=act,
                                            connection_matrix=None)
        self.CA3_CA1.train(epochs=1000,
                           epsilon=0.1,
                           update_offsets=0.0,
                           momentum=0.0,
                           l1norm=0.0,
                           l2norm=0.0,
                           corruptor=[None, None, None])

        self.DG_CA3 = Layer.HeteroAssociator(input_dim=self.DG_dim,
                                             input_mean=self.DG_activity,
                                             output_dim=self.CA3_dim,
                                             act=act,
                                             connection_matrix=None)

        self.CA1_EC = Layer.HeteroAssociator(input_dim=self.CA1_dim,
                                             input_mean=self.CA1_activity,
                                             output_dim=self.EC_dim,
                                             act=act,
                                             connection_matrix=None)

        self.binarize_output = binarize_output

    def store_data_point(self,
                         data_point,
                         epsilon_enc=0.1,
                         epsilon_dec=0.1,
                         disturbance=0.0,
                         iterations=10,
                         l1norm=0.0,
                         l2norm=0.0
                         ):
        """ Stores a new datapoint in the network.

        :param data_point: New datapoint to store.
        :type data_point: numpy array.

        :param epsilon_enc: Learning rate for encoder.
        :type epsilon_enc: float

        :param epsilon_dec: Learning rate for decoder.
        :type epsilon_dec: float

        :param disturbance: Noise factor (0,1) (percentage of neurons change activities).
        :type disturbance: float

        :param iterations: Number of updates performed.
        :type iterations: int

        :param l1norm: L1 weight decay term (0,1)
        :type l1norm: float

        :param l2norm: L2 weight decay term (0,1)
        :type l2norm: float
        """
        data_EC = data_point.reshape(1, self.EC_dim)
        data_DG = self.EC_DG.calculate_output(data_EC, binarize_output=self.binarize_output)
        intrinsic = self.CA3_CA3.get_current_state(binarize_output=self.binarize_output).reshape(1, self.CA3_dim)
        data_CA1 = self.CA3_CA1.calculate_output(intrinsic, binarize_output=self.binarize_output)
        corr = Dropout(disturbance)
        if disturbance <= 0.0:
            corr = None
        self.DG_CA3.store_datapoint(input_data=data_DG,
                                    output_data=intrinsic,
                                    epochs=iterations,
                                    epsilon=epsilon_enc,
                                    update_offsets=0.0,
                                    momentum=0.0,
                                    l1norm=l1norm,
                                    l2norm=l2norm,
                                    corruptor=corr)
        self.CA1_EC.store_datapoint(input_data=data_CA1,
                                    output_data=data_EC,
                                    epochs=iterations,
                                    epsilon=epsilon_dec,
                                    update_offsets=0.0,
                                    momentum=0.0,
                                    l1norm=l1norm,
                                    l2norm=l2norm,
                                    corruptor=corr)
        self.CA3_CA3.get_next_state(binarize_output=self.binarize_output)

    def encode(self, data):
        """ Encodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.DG_CA3.calculate_output(self.EC_DG.calculate_output(data, binarize_output=self.binarize_output),
                                            binarize_output=self.binarize_output)

    def decode(self, data):
        """ Decodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.CA1_EC.calculate_output(self.CA3_CA1.calculate_output(data, binarize_output=self.binarize_output),
                                            binarize_output=self.binarize_output)


class Hippo3ECDGCA3CA1EC(HippoECCA3EC):
    """ Hippocampus experiments with Dentate gyrus and CA1.

    """

    def __init__(self,
                 EC_dim,
                 EC_activity,
                 DG_dim,
                 DG_activity,
                 CA3_dim,
                 CA3_activity,
                 CA3_capacity,
                 CA1_dim,
                 CA1_activity,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 binarize_output=False
                 ):
        """ Constructor.

        :param EC_dim: Dimensions in EC.
        :type EC_dim: int

        :param EC_activity: Average activity of EC data.
        :type EC_activity: numpy array (1,EC_dim) or scalar

        :param DG_dim: Dimensions in DG.
        :type DG_dim: int

        :param DG_activity: Average activity of DG.
        :type DG_activity: numpy array (1,DG_dim) or scalar

        :param CA3_dim: Dimensions in CA3.
        :type EC_activity: int

        :param CA3_activity: Average activity of EC data.
        :type CA3_activity: numpy array (1,CA3_dim) or scalar

        :param CA3_capacity: Number of intrinsic states or sequence length.
        :type CA3_capacity: int

        :param CA1_dim: Dimensions in CA1.
        :type CA1_dim: int

        :param CA1_activity: Average activity of CA1.
        :type CA1_activity: numpy array (1,CA1_dim) or scalar

        :param act: Activation function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type act: numpy array

        :param cost: Cost function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type cost: pydeep.base.costfunction

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool
        """
        self.EC_dim = EC_dim
        self.EC_activity = EC_activity
        self.DG_dim = DG_dim
        self.DG_activity = DG_activity
        self.CA3_dim = CA3_dim
        self.CA3_capacity = CA3_capacity
        self.CA3_activity = CA3_activity
        self.CA1_dim = CA1_dim
        self.CA1_activity = CA1_activity

        intrinsic_sequence = generate_binary_random_sequence(self.CA3_capacity, self.CA3_dim,
                                                             self.CA3_dim * self.CA3_activity)
        self.CA3_CA3 = Layer.PredictiveAssociator(intrinsic_sequence)
        self.CA3_CA3.train(0.1, 0.1, 1000, 10)

        self.DG_CA3 = Layer.AutoAssociator(input_dim=CA3_dim,
                                           input_mean=CA3_activity,
                                           output_dim=DG_dim,
                                           output_mean=DG_activity,
                                           act=act,
                                           connection_matrix=None)

        self.DG_CA3.train(epochs=1000,
                          epsilon=0.05,
                          update_offsets=0.0,
                          momentum=0.0,
                          l1norm=0.0,
                          l2norm=0.0,
                          corruptor=[Dropout(0.1), None, None])

        self.CA3_CA1 = Layer.AutoAssociator(input_dim=CA3_dim,
                                            input_mean=CA3_activity,
                                            output_dim=CA1_dim,
                                            output_mean=CA1_activity,
                                            act=act,
                                            connection_matrix=None)

        self.CA3_CA1.train(epochs=1000,
                           epsilon=0.05,
                           update_offsets=0.0,
                           momentum=0.0,
                           l1norm=0.0,
                           l2norm=0.0,
                           corruptor=[Dropout(0.1), None, None])

        self.EC_DG = Layer.HeteroAssociator(input_dim=self.EC_dim,
                                            input_mean=self.EC_activity,
                                            output_dim=self.DG_dim,
                                            act=act,
                                            sparse=True,
                                            connection_matrix=None)

        self.CA1_EC = Layer.HeteroAssociator(input_dim=self.CA1_dim,
                                             input_mean=self.CA1_activity,
                                             output_dim=self.EC_dim,
                                             act=act,
                                             connection_matrix=None)

        self.binarize_output = binarize_output

    def store_data_point(self,
                         data_point,
                         epsilon_enc=0.1,
                         epsilon_dec=0.1,
                         disturbance=0.0,
                         iterations=10,
                         l1norm=0.0,
                         l2norm=0.0
                         ):
        """ Stores a new datapoint in the network.

        :param data_point: New datapoint to store.
        :type data_point: numpy array.

        :param epsilon_enc: Learning rate for encoder.
        :type epsilon_enc: float

        :param epsilon_dec: Learning rate for decoder.
        :type epsilon_dec: float

        :param disturbance: Noise factor (0,1) (percentage of neurons change activities).
        :type disturbance: float

        :param iterations: Number of updates performed.
        :type iterations: int

        :param l1norm: L1 weight decay term (0,1)
        :type l1norm: float

        :param l2norm: L2 weight decay term (0,1)
        :type l2norm: float
        """
        intrinsic = self.CA3_CA3.get_current_state(binarize_output=self.binarize_output).reshape(1, self.CA3_dim)
        data_EC = data_point.reshape(1, self.EC_dim)
        data_DG = self.DG_CA3.calculate_output(intrinsic, binarize_output=self.binarize_output)
        data_CA1 = self.CA3_CA1.calculate_output(intrinsic, binarize_output=self.binarize_output)
        corr = Dropout(disturbance)
        if disturbance <= 0.0:
            corr = None
        self.EC_DG.store_datapoint(input_data=data_EC,
                                   output_data=data_DG,
                                   epochs=iterations,
                                   epsilon=epsilon_enc,
                                   update_offsets=0.0,
                                   momentum=0.0,
                                   l1norm=l1norm,
                                   l2norm=l2norm,
                                   corruptor=corr)
        self.CA1_EC.store_datapoint(input_data=data_CA1,
                                    output_data=data_EC,
                                    epochs=iterations,
                                    epsilon=epsilon_dec,
                                    update_offsets=0.0,
                                    momentum=0.0,
                                    l1norm=l1norm,
                                    l2norm=l2norm,
                                    corruptor=corr)
        self.CA3_CA3.get_next_state(self.binarize_output)

    def encode(self, data):
        """ Encodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.DG_CA3.calculate_input(self.EC_DG.calculate_output(data, self.binarize_output),
                                           self.binarize_output)

    def decode(self, data):
        """ Decodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.CA1_EC.calculate_output(self.CA3_CA1.calculate_output(data, self.binarize_output),
                                            self.binarize_output)


class HippoECDGCA3ECJointInput(object):
    """ Hippocampus experiments with perforantal pathway EC-CA1
    """

    def __init__(self,
                 EC_dim,
                 EC_activity,
                 DG_dim,
                 DG_activity,
                 CA3_dim,
                 CA3_activity,
                 CA3_capacity,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 binarize_output=False
                 ):
        """ Constructor.

        :param EC_dim: Dimensions in EC.
        :type EC_dim: int

        :param EC_activity: Average activity of EC data.
        :type EC_activity: numpy array (1,EC_dim) or scalar

        :param DG_dim: Dimensions in DG.
        :type DG_dim: int

        :param DG_activity: Average activity of DG.
        :type DG_activity: numpy array (1,DG_dim) or scalar

        :param CA3_dim: Dimensions in CA3.
        :type EC_activity: int

        :param CA3_activity: Average activity of EC data.
        :type CA3_activity: numpy array (1,CA3_dim) or scalar

        :param CA3_capacity: Number of intrinsic states or sequence length.
        :type CA3_capacity: int

        :param act: Activation function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type act: numpy array

        :param cost: Cost function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized Hebb learning.
        :type cost: pydeep.base.costfunction

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool
        """
        self.EC_dim = EC_dim
        self.EC_activity = EC_activity
        self.DG_dim = DG_dim
        self.DG_activity = DG_activity
        self.CA3_dim = CA3_dim
        self.CA3_capacity = CA3_capacity
        self.CA3_activity = CA3_activity

        intrinsic_sequence = generate_binary_random_sequence(self.CA3_capacity, self.CA3_dim,
                                                             self.CA3_dim * self.CA3_activity)
        self.CA3_CA3 = Layer.PredictiveAssociator(intrinsic_sequence)
        self.CA3_CA3.train(0.1, 0.1, 1000, 10)

        self.EC_DG = Layer.AutoAssociator(input_dim=EC_dim,
                                          input_mean=EC_activity,
                                          output_dim=DG_dim,
                                          output_mean=DG_activity,
                                          act=act,
                                          sparse=True,
                                          connection_matrix=None)
        self.EC_DG.train(epochs=1000,
                         epsilon=0.1,
                         update_offsets=0.0,
                         momentum=0.0,
                         l1norm=0.0,
                         l2norm=0.0,
                         corruptor=Dropout(0.1))

        self.ECDG_CA3 = Layer.HeteroAssociator(input_dim=self.DG_dim + self.EC_dim,
                                               input_mean=np.hstack((self.DG_activity * np.ones(self.DG_dim),
                                                                     self.EC_activity * np.ones(self.EC_dim))).reshape(
                                                   1, self.DG_dim + self.EC_dim),
                                               output_dim=self.CA3_dim,
                                               act=act,
                                               connection_matrix=None)

        self.CA3_EC = Layer.HeteroAssociator(input_dim=self.CA3_dim,
                                             input_mean=np.mean(intrinsic_sequence, axis=0).reshape(1, self.CA3_dim),
                                             output_dim=self.EC_dim,
                                             act=act,
                                             connection_matrix=None)

        self.binarize_output = binarize_output

    def store_data_point(self,
                         data_point,
                         epsilon_enc=1.0,
                         epsilon_dec=0.01,
                         disturbance=0.05,
                         iterations=10,
                         l1norm=0.0,
                         l2norm=0.0
                         ):
        """ Stores a new datapoint in the network.

        :param data_point: New datapoint to store.
        :type data_point: numpy array.

        :param epsilon_enc: Learning rate for encoder.
        :type epsilon_enc: float

        :param epsilon_dec: Learning rate for decoder.
        :type epsilon_dec: float

        :param disturbance: Noise factor (0,1) (percentage of neurons change activities).
        :type disturbance: float

        :param iterations: Number of updates performed.
        :type iterations: int

        :param l1norm: L1 weight decay term (0,1)
        :type l1norm: float

        :param l2norm: L2 eeight decay term (0,1)
        :type l2norm: float
        """
        data_EC = data_point.reshape(1, self.EC_dim)
        data_DG = np.hstack((self.EC_DG.calculate_output(data_EC, self.binarize_output), data_EC))
        intrinsic = self.CA3_CA3.get_current_state(self.binarize_output).reshape(1, self.CA3_dim)
        corr = Dropout(disturbance)
        if disturbance <= 0.0:
            corr = None
        self.ECDG_CA3.store_datapoint(input_data=data_DG,
                                      output_data=intrinsic,
                                      epochs=iterations,
                                      epsilon=epsilon_enc,
                                      update_offsets=0.0,
                                      momentum=0.0,
                                      l1norm=l1norm,
                                      l2norm=l2norm,
                                      corruptor=corr)
        self.CA3_EC.store_datapoint(input_data=intrinsic,
                                    output_data=data_EC,
                                    epochs=iterations,
                                    epsilon=epsilon_dec,
                                    update_offsets=0.0,
                                    momentum=0.0,
                                    l1norm=l1norm,
                                    l2norm=l2norm,
                                    corruptor=corr)
        self.CA3_CA3.get_next_state(self.binarize_output)

    def encode_decode(self, data):
        """ Encodes and decodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        output = self.EC_DG.calculate_output(data, self.binarize_output)
        output = np.hstack((output, data))
        output = self.ECDG_CA3.calculate_output(output, self.binarize_output)
        return self.CA3_EC.calculate_output(output, self.binarize_output)

    def encode_next_decode(self, data):
        """ Encodes the data, gets the next intrinsic state and decodes it such that based on x_t , x_t+1 is calculated.

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        output = self.EC_DG.calculate_output(data, self.binarize_output)
        output = np.hstack((output, data))
        output = self.ECDG_CA3.calculate_output(output, self.binarize_output)
        output = self.CA3_CA3.calculate_output(output, self.binarize_output)
        return self.CA3_EC.calculate_output(output, self.binarize_output)

    def full_loop_encode_next_decode(self, data):
        """ Calculates the full loop output. Each given data point is fed into the network, which predicts/outputs the
            next data point, which is fed to the network again. This is repeated for the entire sequence length.

        :param data: Input data, for each data point the full loop is done.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        curr = data
        result = None
        for _ in range(self.CA3_capacity):
            curr = self.encode_next_decode(curr)
            if result is None:
                result = curr
            else:
                result = np.vstack((result, curr))
        return result

    def full_intrinsic_loop_encode_next_decode(self, data):
        """ Calculates the full intrinsic loop output. Each given data point is fed into, which triggers a state in CA3.
            The intrinsic dynamics are then used to reconstruct the full sequence.

        :param data: Input data, for each data point the full loop is done.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        result = None
        curr = self.ECDG_CA3.calculate_output(np.hstack((self.EC_DG.calculate_output(data, self.binarize_output), data))
                                              , self.binarize_output)
        for _ in range(self.CA3_capacity):
            curr = self.CA3_CA3.calculate_output(curr, self.binarize_output)
            output = self.CA3_EC.calculate_output(curr, self.binarize_output)
            if result is None:
                result = output
            else:
                result = np.vstack((result, output))
        return result


class HippoECDGCA3ECNotIntrinsic(HippoECCA3EC):
    """ Hippocampus experiments without Intrinsic sequence.
    """

    def __init__(self,
                 EC_dim,
                 EC_activity,
                 DG_dim,
                 DG_activity,
                 CA3_dim,
                 CA3_activity,
                 CA3_capacity,
                 act=HebbSigmoid,
                 cost=SquaredError,
                 binarize_output=False
                 ):
        """ Constructor.

        :param EC_dim: Dimensions in EC.
        :type EC_dim: int

        :param EC_activity: Average activity of EC data.
        :type EC_activity: numpy array (1,EC_dim) or scalar

        :param DG_dim: Dimensions in DG.
        :type DG_dim: int

        :param DG_activity: Average activity of DG.
        :type DG_activity: numpy array (1,DG_dim) or scalar

        :param CA3_dim: Dimensions in CA3.
        :type EC_activity: int

        :param CA3_activity: Average activity of EC data.
        :type CA3_activity: numpy array (1,CA3_dim) or scalar

        :param CA3_capacity: Number of intrinsic states or sequence length.
        :type CA3_capacity: int

        :param act: Activation function, if single layer + Squared error + HebbSigmoid is chosen it performaes
                    generlized Hebb learning.
        :type act: numpy array

        :param cost: Cost function, if single layer + Squared error + HebbSigmoid is chosen it performaes generlized
                     Hebb learning.
        :type cost: pydeep.base.costfunction

        :param binarize_output: If True the output gets binarized, i.e. x in {0,1}^N
        :type binarize_output: bool
        """
        self.EC_dim = EC_dim
        self.EC_activity = EC_activity
        self.DG_dim = DG_dim
        self.DG_activity = DG_activity
        self.CA3_dim = CA3_dim
        self.CA3_capacity = CA3_capacity
        self.CA3_activity = CA3_activity

        self.EC_DG = Layer.AutoAssociator(input_dim=EC_dim,
                                          input_mean=EC_activity,
                                          output_dim=DG_dim,
                                          output_mean=DG_activity,
                                          act=act,
                                          sparse=True,
                                          connection_matrix=None)
        self.EC_DG.train(epochs=1000,
                         epsilon=0.1,
                         update_offsets=0.0,
                         momentum=0.0,
                         l1norm=0.0,
                         l2norm=0.0,
                         corruptor=[Dropout(0.1), None, None])

        self.DG_CA3 = Layer.AutoAssociator(input_dim=DG_dim,
                                           input_mean=DG_activity,
                                           output_dim=CA3_dim,
                                           output_mean=CA3_activity,
                                           act=act,
                                           connection_matrix=None)
        self.DG_CA3.train(epochs=1000,
                          epsilon=0.05,
                          update_offsets=0.0,
                          momentum=0.0,
                          l1norm=0.0,
                          l2norm=0.0,
                          corruptor=[None, None, None])

        self.CA3_EC = Layer.AutoAssociator(input_dim=CA3_dim,
                                           input_mean=CA3_activity,
                                           output_dim=EC_dim,
                                           output_mean=EC_activity,
                                           act=act,
                                           connection_matrix=None)
        self.CA3_EC.train(epochs=1000,
                          epsilon=0.05,
                          update_offsets=0.0,
                          momentum=0.0,
                          l1norm=0.0,
                          l2norm=0.0,
                          corruptor=[Dropout(0.05), None, None])

        self.CA3_CA3 = Layer.HeteroAssociator(input_dim=self.CA3_dim,
                                              input_mean=self.CA3_activity,
                                              output_dim=self.CA3_dim,
                                              act=act,
                                              connection_matrix=None)

        self.binarize_output = binarize_output

    def store_data_point(self,
                         data_point1,
                         data_point2,
                         epsilon_enc=0.1,
                         epsilon_dec=0.1,
                         disturbance=0.0,
                         iterations=10,
                         l1norm=0.0,
                         l2norm=0.0
                         ):
        """ Stores a new datapoint in the network.

        :param data_point: New datapoint to store.
        :type data_point: numpy array.

        :param epsilon_enc: Learning rate for encoder.
        :type epsilon_enc: float

        :param epsilon_dec: Learning rate for decoder.
        :type epsilon_dec: float

        :param disturbance: Noise factor (0,1) (percentage of neurons change activities).
        :type disturbance: float

        :param iterations: Number of updates performed.
        :type iterations: int

        :param l1norm: L1 weight decay term (0,1)
        :type l1norm: float

        :param l2norm: L2 eeight decay term (0,1)
        :type l2norm: float
        """
        corr = Dropout(disturbance)
        if disturbance <= 0.0:
            corr = None
        self.CA3_CA3.store_datapoint(input_data=self.encode(data_point1),
                                     output_data=self.encode(data_point2),
                                     epochs=iterations,
                                     epsilon=epsilon_enc,
                                     update_offsets=0.0,
                                     momentum=0.0,
                                     l1norm=l1norm,
                                     l2norm=l2norm,
                                     corruptor=corr)

    def encode(self, data):
        """ Encodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.DG_CA3.calculate_output(self.EC_DG.calculate_output(data, self.binarize_output),
                                            self.binarize_output)

    def decode(self, data):
        """ Decodes the data (reconstruction).

        :param data: Input data.
        :type data: numpy array

        :return: Output for the given data.
        :rtype: numpy array (2D)

        """
        return self.CA3_EC.calculate_output(data, self.binarize_output)
'''