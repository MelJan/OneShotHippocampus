''' This module provides restricted Boltzmann machines (RBMs) with different 
    types of units. The structure is very close to the mathematical derivations
    to simplify the understanding. In addition, the modularity helps to create
    other kind of RBMs without adapting the training algorithms. 

    :Implemented:
        - centered BinaryBinary RBM (BB-RBM)
        - centered GaussianBinary RBM (GB-RBM) with fixed variance
        - centered GaussianBinaryVariance RBM (GB-RBM) with trainable variance

    :Info: 
        For the derivations .. seealso::
        https://www.ini.rub.de/PEOPLE/wiskott/Reprints/Melchior-2012-MasterThesis-RBMs.pdf
        
        A usual way to create a new unit is to inherit from a given RBM class
        and override the functions that changed, e.g. Gaussian-Binary RBM 
        inherited from the Binary-Binary RBM. 
   
    :Version:
        3.0
        
    :Date:
        11.02.2014
    
    :Author:
        Jan Melchior
        
    :Contact:
        pyrbm.info@gmail.com
        
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
from pydeep.misc.activationfunction import Sigmoid
from pydeep.misc.basicstructures import BipartiteGraph

class BinaryBinaryRBM(BipartiteGraph):
    ''' Implementation of a centered restricted Boltzmann machine with binary
        visible and binary hidden units.     

    '''
  
    def __init__(self, 
                  number_visibles, 
                  number_hiddens,  
                  data = None, 
                  initial_weights = 'AUTO', 
                  initial_visible_bias = 'AUTO', 
                  initial_hidden_bias = 'AUTO', 
                  initial_visible_offsets = 'AUTO', 
                  initial_hidden_offsets = 'AUTO', 
                  dtype = numx.float64):
        ''' This function initializes all necessary parameters and data 
            structures. It is recommended to pass the training data to 
            initialize the network automatically.
            
        :Parameters:
            number_visibles:         Number of the visible variables.
                                    -type: int
                                  
            number_hiddens           Number of hidden variables.
                                    -type: int
                                  
            data:                    The training data for parameter 
                                     initialization if 'AUTO' is chosen 
                                     for the corresponding parameter.
                                    -type: None or 
                                           numpy array [num samples, input dim]
                                        
            initial_weights:         Initial weights.
                                     'AUTO' and a scalar are random init.
                                    -type: 'AUTO', scalar or 
                                          numpy array [input dim, output_dim]
                                  
            initial_visible_bias:    Initial visible bias.
                                     'AUTO' is random, 'INVERSE_SIGMOID' is the
                                     inverse Sigmoid of the visilbe mean.
                                     If a scalar is passed all values are 
                                     initialized with it. 
                                    -type: 'AUTO','INVERSE_SIGMOID', scalar or 
                                          numpy array [1, input dim]
                                  
            initial_hidden_bias:     Initial hidden bias.
                                     'AUTO' is random, 'INVERSE_SIGMOID' is the
                                     inverse Sigmoid of the hidden mean.
                                     If a scalar is passed all values are 
                                     initialized with it. 
                                    -type: 'AUTO','INVERSE_SIGMOID', scalar or 
                                          numpy array [1, output_dim]
                                  
            initial_visible_offsets: Initial visible offset values.
                                     AUTO=data mean or 0.5 if no data is given.
                                     If a scalar is passed all values are 
                                     initialized with it. 
                                    -type: 'AUTO', scalar or 
                                           numpy array [1, input dim]
                                  
            initial_hidden_offsets:  Initial hidden offset values.
                                     AUTO = 0.5
                                     If a scalar is passed all values are 
                                     initialized with it. 
                                    -type: 'AUTO', scalar or 
                                           numpy array [1, output_dim]
                        
            dtype:                   Used data type i.e. numpy.float64
                                    -type: numpy.float32 or numpy.float64 or 
                                           numpy.float128  
            
        '''
        
        # Minimal temperature for the model
        self._MIN_TEMP = 0.0
        
        # Call constructor of superclass
        super(BinaryBinaryRBM, 
              self).__init__(number_visibles = number_visibles, 
                             number_hiddens = number_hiddens, 
                             data = data,
                             initial_weights = initial_weights,
                             initial_visible_bias = initial_visible_bias,
                             initial_hidden_bias = initial_hidden_bias,
                             initial_visible_offsets = initial_visible_offsets,
                             initial_hidden_offsets = initial_hidden_offsets,
                             dtype = dtype)

    def _calculate_weight_gradient(self, v, h):
        ''' This function calculates the gradient for the weights from the      
            visible and hidden activations.
        
        :Parameters:
            v: Visible activations.
              -type: numpy arrays [batchsize, input dim]
                                
            h: Hidden activations.
              -type: numpy arrays [batchsize, output dim]
                                
        :Returns:
            Weight gradient.
           -type: numpy arrays [input dim, output dim]
            
        ''' 
        return numx.dot((v - self.ov).T , h - self.oh)

    def _calculate_visible_bias_gradient(self, v):
        ''' This function calculates the gradient for the visible biases.
        
        :Parameters:
            v: Visible activations.
              -type: numpy arrays [batch_size, input dim]
                                
        :Returns:
            Visible bias gradient.
           -type: numpy arrays [1, input dim]
               
        '''       
        return numx.sum(v - self.ov, axis=0).reshape(1,v.shape[1])
    
    def _calculate_hidden_bias_gradient(self, h):
        ''' This function calculates the gradient for the hidden biases.
        
        :Parameters:                      
            h:  Hidden activations.
               -type: numpy arrays [batch size, output dim]
                                
        :Returns:
            Hidden bias gradient.
           -type: numpy arrays [1, output dim]
            
        ''' 
        return numx.sum(h - self.oh, axis=0).reshape(1, h.shape[1])
    
    def calculate_gradients(self, v, h):
        ''' This function calculates all gradients of this RBM and returns 
            them as a list of arrays. This keeps the flexibility of adding 
            parameters which will be updated by the training algorithms.
        
        :Parameters:
            v: Visible activations.
              -type: numpy arrays [batch size, output dim]
                                
            h: Hidden activations.
              -type: numpy arrays [batch size, output dim]
                                
        :Returns:
            Gradients for all parameters.
           -type: list of numpy arrays (num parameters x [parameter.shape])

        ''' 
        return [self._calculate_weight_gradient(v, h)
                ,self._calculate_visible_bias_gradient(v)
                ,self._calculate_hidden_bias_gradient(h)]    

    def sample_v(self, v, beta=1.0):
        ''' Samples the visible variables from the 
            conditional probabilities v given h.  
        
        :Parameters:
            v:  Conditional probabilities of v given h.
               -type: numpy array [batch size, input dim]

        :Returns: 
            States for v.
           -type: numpy array [batch size, input dim]
        
        '''
        return self.dtype(v > numx.random.random(v.shape))
                                                        
    def sample_h(self, h, beta=1.0):
        ''' Samples the hidden variables from the 
            conditional probabilities h given v.
        
        :Parameters:
            ph: Conditional probabilities of h given v.
               -type: numpy array [batch size, output dim]
                           
        :Returns: 
            States for h.
           -type: numpy array [batch size, output dim]
            
        '''
        return self.dtype(h> numx.random.random(h.shape))
    
    def probability_v_given_h(self, h, beta=1.0, AIS = False):
        ''' Calculates the conditional probabilities of v given h.      
        
        :Parameters:
            h:    Hidden states.
                 -type: numpy array [batch size, output dim]
                   
            beta: Allows to sample from a given inverse temperature 
                  beta, or if a vector is given to sample from 
                  different betas simultaneously.
                 -type: float or numpy array [batch size, 1]

            AIS:  Flag for Annealed importance sampling:
                  If true onlz the weights are scaled by beta:
                 -type: bool

        :Returns: 
            Conditional probabilities v given h.
           -type: numpy array [batch size, input dim]
        
        '''
        activation = numx.dot(h-self.oh, self.w.T)
        if AIS==True:
            activation *= beta
            activation += self.bv
        else:
            activation += self.bv
            activation *= beta
        return Sigmoid.f(activation)
    
    def probability_h_given_v(self, v, beta=1.0,AIS = False):
        ''' Calculates the conditional probabilities of h given v.      
        
        :Parameters:
            v:    Visible states.
                 -type: numpy array [batch size, input dim]
                   
            beta: Allows to sample from a given inverse temperature 
                  beta, or if a vector is given to sample from 
                  different betas simultaneously.
                 -type: float or numpy array [batch size, 1]
                 
            AIS:  Flag for Annealed importance sampling:
                  If true onlz the weights are scaled by beta:
                 -type: bool

        :Returns: 
            Conditional probabilities h given v.
           -type: numpy array [batch size, output dim]
        
        '''
        activation = numx.dot(v-self.ov, self.w)
        if AIS==True:
            activation *= beta
            activation += self.bh
        else:
            activation += self.bh
            activation *= beta
        return Sigmoid.f(activation)
        
    def energy(self, v, h, beta=1.0):
        ''' Compute the energy of the RBM given observed variable states v
            and hidden variables state h.
        
        :Parameters:
            v:    Visible states.
                 -type: numpy array [batch size, input dim]
            
            h:    Hidden states.
                 -type: numpy array [batch size, output dim]
                   
            beta: Allows to sample from a given inverse temperature 
                  beta, or if a vector is given to sample from 
                  different betas simultaneously.
                 -type: float or numpy array [batch size, 1]
                  
        :Returns:
            Energy of v and h.
           -type: numpy array [batch size,1]
        
        '''
        return beta * (- numx.dot(v-self.ov, self.bv.T) 
                        - numx.dot(h-self.oh, self.bh.T) 
                        - numx.sum(numx.dot(v-self.ov, self.w) * (h-self.oh) 
                                   ,axis=1).reshape(v.shape[0], 1))

    def unnormalized_log_probability_v(self, v, beta = 1.0, AIS = False):
        ''' Computes the unnormalized log probabilities of v.
        
        :Parameters:
            v:    Visible states.
                 -type: numpy array [batch size, input dim]
                  
            beta: Allows to sample from a given inverse temperature 
                  beta, or if a vector is given to sample from 
                  different betas simultaneously.
                 -type: float or numpy array [batch size, 1]
                 
            AIS:  Flag for Annealed importance sampling:
                  If true onlz the weights are scaled by beta:
                 -type: bool

        :Returns:
            Unnormalized log probability of v.
           -type: numpy array [batch size, 1]
            
        '''    
        activation = numx.dot(v - self.ov, self.w)
        bias = numx.dot(v-self.ov, self.bv.T).reshape(v.shape[0], 1) 
        if AIS==True:
            activation *= beta
            activation += self.bh
        else:
            activation += self.bh
            activation *= beta
            bias *= beta
        return bias + numx.sum(
                               numx.log(
                                        numx.exp(activation*(1 - self.oh)) 
                                      + numx.exp(-activation*self.oh)) 
                               , axis=1).reshape(v.shape[0], 1)
                                     
    def unnormalized_log_probability_h(self, h, beta=1.0):
        ''' Computes the unnormalized log probabilities of h.
        
        :Parameters:
            h:    Hidden states.
                 -type: numpy array [batch size, input dim]
                  
            beta: Allows to sample from a given inverse temperature 
                  beta, or if a vector is given to sample from 
                  different betas simultaneously.
                 -type: float or numpy array [batch size, 1]
            
        :Returns:
            Unnormalized log probability of h.
           -type: numpy array [batch size, 1]
            
        '''
        activation = beta * (numx.dot(h - self.oh, self.w.T) + self.bv)
        bias = beta * numx.dot(h - self.oh, (self.bh).T).reshape(h.shape[0], 1)
        return bias + numx.sum(
                               numx.log(
                                        numx.exp(activation*(1 - self.ov)) 
                                      + numx.exp(-activation*self.ov)) 
                               , axis=1).reshape(h.shape[0], 1)

    def log_probability_v(self, logZ, v, beta=1.0):
        ''' Computes the log-probability / LogLikelihood(LL) for the given 
            visible units for this model. 
            To estimate the LL we need to know the logarithm of the partition 
            function Z. For small models it is possible to calculate Z, 
            however since this involves calculating all possible hidden 
            states, it is intractable for bigger models. As an estimation 
            method annealed importance sampling (AIS) can be used instead.
        
        :Parameters:
            logZ: The logarithm of the partition function.
                 -type: float
            
            v:    Visible states.
                 -type: numpy array [batch size, input dim]
            
            beta: Allows to sample from a given inverse temperature 
                  beta, or if a vector is given to sample from 
                  different betas simultaneously.
                 -type: float or numpy array [batch size, 1]
            
        :Returns:
            Log probability for visible_states.
           -type: numpy array [batch size, 1]
            
        '''
        return self.unnormalized_log_probability_v(v, beta) - logZ

    def log_probability_h(self, logZ,  h, beta=1.0):
        ''' Computes the log-probability / LogLikelihood(LL) for the given 
            hidden units for this model. 
            To estimate the LL we need to know the logarithm of the partition 
            function Z. For small models it is possible to calculate Z, 
            however since this involves calculating all possible hidden 
            states, it is intractable for bigger models. As an estimation 
            method annealed importance sampling (AIS) can be used instead.
        
        :Parameters:
            logZ: The logarithm of the partition function.
                 -type: float
            
            h:    Hidden states.
                 -type: numpy array [batch size, output dim]
            
            beta: Allows to sample from a given inverse temperature 
                  beta, or if a vector is given to sample from 
                  different betas simultaneously.
                 -type: float or numpy array [batch size, 1]
            
        :Returns:
            Log probability for hidden_states.
           -type: numpy array [batch size, 1]
            
        '''
        return self.unnormalized_log_probability_h(h, beta) - logZ

    def log_probability_v_h(self, logZ, v, h, beta = 1.0):
        ''' Computes the joint log-probability / LogLikelihood(LL) for the 
            given visible and hidden units for this model. 
            To estimate the LL we need to know the logarithm of the partition 
            function Z. For small models it is possible to calculate Z, 
            however since this involves calculating all possible hidden 
            states, it is intractable for bigger models. As an estimation 
            method annealed importance sampling (AIS) can be used instead.
         
        :Parameters:
            logZ: The logarithm of the partition function.
                 -type: float
            
            v:    Visible states.
                 -type: numpy array [batch size, input dim]
                 
            h:    Hidden states.
                 -type: numpy array [batch size, output dim]
            
            beta: Allows to sample from a given inverse temperature 
                  beta, or if a vector is given to sample from 
                  different betas simultaneously.
                 -type: float or numpy array [batch size, 1]
            
        :Returns:
            Joint log probability for v and h.
           -type: numpy array [batch size, 1]
            
        '''
        return self.energy(v, h, beta) - logZ

    def _base_log_partition(self):
        ''' Returns the base partition function which needs to be 
            calculateable. 
                                      
        :Returns:
            Partition function for zero parameters.
           -type: float
        
        '''
        return (numx.sum(numx.log(1.0+ numx.exp(self.bv)))
               +numx.sum(numx.log(1.0+numx.exp(self.bh))))

class GaussianBinaryRBM(BinaryBinaryRBM):
    ''' Implementation of a centered Restricted Boltzmann machine with Gaussian
        visible and binary hidden units.
    
    '''
    
    def __init__(self, 
                  number_visibles, 
                  number_hiddens, 
                  data=None, 
                  initial_weights='AUTO', 
                  initial_visible_bias='AUTO', 
                  initial_hidden_bias='AUTO', 
                  initial_sigma='AUTO', 
                  initial_visible_offsets=0.0, 
                  initial_hidden_offsets=0.0, 
                  dtype=numx.float64):
        ''' This function initializes all necessary parameters and data 
            structures. It is recommended to pass the training data to 
            initialize the network automatically.
            
        :Parameters:
            number_visibles:         Number of the visible variables.
                                    -type: int
                                  
            number_hiddens           Number of hidden variables.
                                    -type: int
                                  
            data:                    The training data for parameter 
                                     initialization if 'AUTO' is chosen 
                                     for the corresponding parameter.
                                    -type: None or 
                                           numpy array [num samples, input dim]
                                        
            initial_weights:         Initial weights.
                                     'AUTO' and a scalar are random init.
                                    -type: 'AUTO', scalar or 
                                          numpy array [input dim, output_dim]
                                  
            initial_visible_bias:    Initial visible bias.
                                     'AUTO' is random, 'INVERSE_SIGMOID' is the
                                     inverse Sigmoid of the visilbe mean.
                                     If a scalar is passed all values are 
                                     initialized with it. 
                                    -type: 'AUTO','INVERSE_SIGMOID', scalar or 
                                          numpy array [1, input dim]
                                  
            initial_hidden_bias:     Initial hidden bias.
                                     'AUTO' is random, 'INVERSE_SIGMOID' is the
                                     inverse Sigmoid of the hidden mean.
                                     If a scalar is passed all values are 
                                     initialized with it. 
                                    -type: 'AUTO','INVERSE_SIGMOID', scalar or 
                                          numpy array [1, output_dim]

            initial_sigma:           Initial standard deviation for the model.
                                    -type: 'AUTO', scalar or 
                                           numpy array [1, input_dim]

            initial_visible_offsets: Initial visible offset values.
                                     AUTO=data mean or 0.5 if no data is given.
                                     If a scalar is passed all values are 
                                     initialized with it. 
                                    -type: 'AUTO', scalar or 
                                           numpy array [1, input dim]
                                  
            initial_hidden_offsets:  Initial hidden offset values.
                                     AUTO = 0.5
                                     If a scalar is passed all values are 
                                     initialized with it. 
                                    -type: 'AUTO', scalar or 
                                           numpy array [1, output_dim]
                        
            dtype:                   Used data type i.e. numpy.float64
                                    -type: numpy.float32 or numpy.float64 or 
                                           numpy.float128  
            
        '''

        if isinstance(data,list):
            data = numx.concatenate(data)
        
        if initial_visible_bias is 'AUTO' and data != None:
            initial_visible_bias = numx.mean(data, axis=0).reshape(1,
                                                           self.input_dim)
        
        if initial_visible_offsets is 'AUTO':
            initial_visible_offsets = numx.zeros((1, self.input_dim), 
                                                 dtype=dtype)
        
        # Call constructor of superclass
        super(GaussianBinaryRBM, 
              self).__init__(number_visibles = number_visibles, 
                             number_hiddens = number_hiddens, 
                             data = data,
                             initial_weights = initial_weights,
                             initial_visible_bias = initial_visible_bias,
                             initial_hidden_bias = initial_hidden_bias,
                             initial_visible_offsets = initial_visible_offsets,
                             initial_hidden_offsets = initial_hidden_offsets,
                             dtype = dtype)
              
        # No Simoid units lead to 4 times smaller initial values
        if initial_weights is 'AUTO':
            self.w /= 4.0

        if initial_sigma is 'AUTO':
            if data == None:
                self.sigma = numx.ones((1, self.input_dim), dtype=dtype)
            else:
                self.sigma = numx.std(data, axis=0
                                      ).reshape(1, self.input_dim)
        else:
            if(numx.isscalar(initial_sigma)):
                self.sigma = numx.ones((1, self.input_dim), 
                                       dtype=dtype) * initial_sigma
            else:
                self.sigma = numx.array(initial_sigma, dtype=dtype)

    def _add_visible_units(self, 
                              num_new_visibles, 
                              position=0, 
                              initial_weights='AUTO', 
                              initial_bias='AUTO', 
                              initial_sigma=1.0,
                              initial_mean = 0.0): 
        ''' This function adds new visible units at teh given position to the 
            model.
            .. Warning:: If the parameters are changed. the trainer needs to be
                     reinitialized.
        
        :Parameters:
            num_new_visibles: The number of new hidden units to add
                             -type: int

            position:         Position where the units should be added.
                             -type: int

            initial_weights:  The initial weight values for the hidden units. 
                             -type: 'AUTO' or 
                                    numpy array [num_new_visibles, output_dim]

            initial_bias:     The initial hidden bias values. 
                             -type: 'AUTO' or numpy array[1, num_new_visibles]
                             
            initial_sigma:    The initial standard deviation for the model.
                             -type: 'AUTO', scalar or 
                                    numpy array [1, num_new_visibles]
                                 
            initial_mean:     The initial visible mean values. 
                             -type: 'AUTO' or 
                                    numpy array [1, num_new_visibles]
                                        
        ''' 
        
        if initial_weights is 'AUTO':
            initial_weights = numx.array((2.0 * numx.random.rand(
                                    num_new_visibles,self.output_dim) - 1.0)
                                  * (numx.sqrt(6.0 / (self.input_dim 
                                    + self.output_dim))), dtype=self.dtype)
            

        super(GaussianBinaryRBM, self)._add_visible_units(num_new_visibles,
                                                          position, 
                                                          initial_weights, 
                                                          initial_bias, 
                                                          initial_mean)
            
        if initial_sigma is 'AUTO':
            new_sigma = numx.ones((1, num_new_visibles), dtype=self.dtype)
        else:
            if(numx.isscalar(initial_sigma)):
                new_sigma = numx.ones((1, num_new_visibles), 
                                      dtype=self.dtype) * initial_sigma
            else:
                new_sigma = numx.array(initial_sigma, dtype=self.dtype)
        self.sigma = numx.insert(self.sigma, numx.ones((num_new_visibles))
                                 *position, new_sigma, axis=1)

    def _add_hidden_units(self, 
                             num_new_hiddens, 
                             position = 0, 
                             initial_weights='AUTO', 
                             initial_bias='AUTO', 
                             initial_mean = 0.0):
        ''' This function adds new hidden units at the given position to the
            model.
            .. Warning:: If the parameters are changed. the trainer needs to be
                     reinitialized.
        
        :Parameters:
            num_new_hiddens: The number of new hidden units to add.
                            -type: int
                            
            position:        Position where the units should be added.
                            -type: int

            initial_weights: The initial weight values for the hidden units. 
                            -type: 'AUTO' or scalar or
                                    numpy array [input_dim, num_new_hiddens]

            initial_bias:    The initial hidden bias values. 
                            -type: 'AUTO' or scalar or 
                                    numpy array [1, num_new_hiddens]
                            
            initial_mean:    The initial hidden mean values. 
                            -type: 'AUTO' or scalar or 
                                    numpy array [1, num_new_hiddens]
                             
        ''' 
        if initial_weights is 'AUTO':
            initial_weights = numx.array((2.0 * numx.random.rand(
                                        self.input_dim, num_new_hiddens) - 1.0)
                                         * ( numx.sqrt(6.0 / (self.input_dim 
                                                          + self.output_dim)))
                                     ,dtype=self.dtype)
        super(GaussianBinaryRBM, self)._add_hidden_units(num_new_hiddens,
                                                         position, 
                                                         initial_weights, 
                                                         initial_bias, 
                                                         initial_mean)
            
    def _remove_visible_units(self, indices):
        ''' This function removes the visible units whose indices are given.
            .. Warning:: If the parameters are changed. the trainer needs to be
                     reinitialized.
        :Parameters:
            indices: Indices to remove.
                    -type: int or list of int or numpy array of int
            
        ''' 
        super(GaussianBinaryRBM, self)._remove_visible_units(indices)
        self.sigma = numx.delete(self.sigma, numx.array(indices), axis=1)  

    def _calculate_weight_gradient(self, v, h):
        ''' This function calculates the gradient for the weights from the 
            hidden and visible activation.
        
        :Parameters:
            v: States of the visible variables.
              -type: numpy arrays [batchsize, input dim]
                                
            h: Probabilities of the hidden variables.
              -type: numpy arrays [batchsize, output dim]
                                
        :Returns:
            Weight gradient.
           -type: numpy arrays [input dim, output dim]
            
        ''' 
        return numx.dot(((v - self.ov)/(self.sigma*self.sigma)).T, h - self.oh)

    def _calculate_visible_bias_gradient(self, v):
        ''' This function calculates the gradient for the visible biases.
        
        :Parameters:
            v: States of the visible variables.
              -type: numpy arrays [batch_size, input dim]
                                
        :Returns:
            visible bias gradient.
           -type: numpy arrays [1, input dim]
            
        '''
        return (numx.sum(v - self.ov - self.bv, axis=0).reshape(1, v.shape[1])
                ) / (self.sigma*self.sigma)
                                    
    def sample_v(self, v, beta = 1.0):
        ''' Samples the visible variables from the conditional probabilities 
            of v given h.  
        
        :Parameters:
            v :   Conditional probabilities of v given h.
                  -type: numpy array [batch size, input dim]

            beta:  Allows to sample from a given inverse temperature
                   beta, or if a vector is given to sample from 
                   different betas simultaneously.
                   .. Warning:: Zero inverse temperatures cause devision
                   by zero errors.
                  -type: float or numpy array [batch size, 1]

        :Returns: 
            States for v.
           -type: numpy array [batch size, input dim]
        
        '''
        return (v + numx.random.randn(v.shape[0], v.shape[1]) 
                * self.sigma/numx.sqrt(beta))
      
    def probability_v_given_h(self, h, beta = 1.0, AIS = False):
        ''' Calculates the conditional probabilities v given h.      
        
        :Parameters:
            h:    Hidden states.
                 -type: numpy array [batch size, output dim]
                   
            beta: Dummy variable for GB-RBMs beta effects the variance 
                  during sampling.
                 -type: float or numpy array [batch size, 1]

            AIS:  Flag for Annealed importance sampling:
                  If true onlz the weights are scaled by beta:
                 -type: bool
    
        :Returns: 
            Conditional probabilities v given h.
           -type: numpy array [batch size, input dim]
        
        '''
        activation = numx.dot(h-self.oh, self.w.T)
        if AIS==True:
            activation *= beta
            activation += self.bv + self.ov
        else:
            activation += self.bv + self.ov
            activation *= beta
        return activation

    def probability_h_given_v(self, v, beta = 1.0, AIS = False):
        ''' Calculates the conditional probabilities h given v.      
        
        :Parameters:
            v:    Visible states / data.
                 -type: numpy array [batch size, input dim]
                   
            beta: Allows to sample from a given inverse temperature beta, or 
                  if a vector is given to sample from different betas 
                  simultaneously.
                 -type: float or numpy array [batch size, 1]

            AIS:  Flag for Annealed importance sampling:
                  If true onlz the weights are scaled by beta:
                 -type: bool

        :Returns: 
            Conditional probabilities h given v.
           -type: numpy array [batch size, output dim]
        
        '''
        activation = numx.dot((v-self.ov) / (self.sigma*self.sigma), self.w)
        if AIS==True:
            activation *= beta
            activation += self.bh
        else:
            activation += self.bh
            activation *= beta
        return Sigmoid.f(activation)

    def energy(self, v, h, beta=1.0):
        ''' Compute the energy of the RBM given observed variable states v 
            and hidden variables state h.
        
        :Parameters:
            v:     The data/visible units states.
                  -type: numpy array [batch size, input dim]
            
            h:     The hidden units states.
                  -type: numpy array [batch size, output dim]
            
            betas: Allows to calculate the energy for given inverse 
                   temperature beta.
                  -type: float
                  
        :Returns:
            Energy of v and h.
           -type: numpy array [batch size,1]
        
        '''
        return beta*(0.5*numx.sum(((v - self.ov - self.bv) / self.sigma) ** 2, 
                                  axis=1).reshape(h.shape[0],1)
               - numx.dot(h-self.oh, self.bh.T) 
               - numx.sum(numx.dot((v - self.ov)/(self.sigma**2), self.w) 
                        * (h-self.oh),axis=1).reshape(h.shape[0],1))
         
    def unnormalized_log_probability_v(self, v, beta = 1.0, AIS = False):
        ''' Computes the unnormalized probability 
            ln(Z*P(v)) = ln(P(v))-ln(Z)+ln(Z) = ln(P(v)).
        
        :Parameters:
            v:    Visible data.
                 -type: numpy array [batch size, input dim]
                  
            beta: Allows to calculate the unnormalized log probability for a 
                  given inverse temperature beta.
                 -type: float

            AIS:  Flag for Annealed importance sampling:
                  If true onlz the weights are scaled by beta:
                 -type: bool

        :Returns:
            Unnormalized log probability of v.
           -type: numpy array [batch size, 1]
            
        '''   
        activation = numx.dot((v-self.ov)/(self.sigma**2),self.w) 
        bias = ((v - self.ov- self.bv) / self.sigma)** 2
        if AIS==True:
            activation *= beta
            activation += self.bh
        else:
            activation += self.bh
            activation *= beta
            bias *= beta
        bias = -0.5*numx.sum(bias, axis=1).reshape(v.shape[0],1)
        return bias + numx.sum(
                               numx.log(
                                        numx.exp(activation*(1 - self.oh)) 
                                      + numx.exp(-activation*self.oh)) 
                               , axis=1).reshape(v.shape[0], 1)         

                         
    def unnormalized_log_probability_h(self, h, beta = 1.0):
        ''' Computes the unnormalized probability 
        
        :Parameters:
            h:    Hidden data. 
                 -type: numpy array [batch size, output dim]
                  
            beta: Allows to calculate the unnormalized log probability for a 
                  given inverse temperature beta.
                 -type: float
            
        :Returns:
            Unnormalized log probability of h.
           -type: numpy array [batch size, 1]
            
        '''
        return ( self.input_dim*0.5*numx.log(2.0*numx.pi)
                 + numx.sum(numx.log(self.sigma)-numx.log(beta)) 
                 + beta*numx.dot(h-self.oh, self.bh.T).reshape(h.shape[0],1) 
                 + numx.sum(beta*((self.bv + numx.dot(h-self.oh, self.w.T)
                                 )/(numx.sqrt(2)*self.sigma))** 2
                          ,axis=1).reshape(h.shape[0],1)\
                 - numx.sum(beta*((self.bv)/ (numx.sqrt(2)*self.sigma)) ** 2))

    def _base_log_partition(self):
        ''' Returns the base partition function which needs to be 
            calculateable. Here W,bv,bh are zero, sigma not!
            Only used for AIS! (CHECKED)
        
        :Returns:
            Partition function for zero parameters.
           -type: float
        
        '''      
        return ( self.input_dim * 0.5 * numx.log(2.0*numx.pi) 
               + numx.sum(numx.log(self.sigma)) 
               + numx.sum(numx.log(1.0+numx.exp(self.bh))))

class GaussianBinaryVarianceRBM(GaussianBinaryRBM):
    ''' Implementation of a Restricted Boltzmann machine with Gaussian visible
        having trainable variances and binary hidden units.
    
    '''
        
    def __init__(self, 
                  number_visibles, 
                  number_hiddens, 
                  data=None, 
                  initial_weights='AUTO', 
                  initial_visible_bias='AUTO', 
                  initial_hidden_bias='AUTO', 
                  initial_sigma='AUTO', 
                  initial_visible_offsets=0.0, 
                  initial_hidden_offsets=0.0,
                  dtype=numx.float64):
        ''' This function initializes all necessary parameters and data 
            structures. See comments for automatically chosen values.
            
        :Parameters:
            number_visibles:         Number of the visible variables.
                                    -type: int
                                  
            number_hiddens           Number of hidden variables.
                                    -type: int
                                  
            data:                    The training data for initializing the 
                                     visible bias.
                                    -type: None or 
                                           numpy array [num samples, input dim]
                                           or List of numpy arrays
                                           [num samples, input dim]
            
            initial_weights:         Initial weights.
                                    -type: 'AUTO', scalar or 
                                           numpy array [input dim, output_dim]
                                  
            initial_visible_bias:    Initial visible bias.
                                    -type: 'AUTO', scalar or 
                                           numpy array [1,input dim]
                                  
            initial_hidden_bias:     Initial hidden bias.
                                    -type: 'AUTO', scalar or 
                                           numpy array [1, output_dim]
                                  
            initial_sigma:           Initial standard deviation for the model.
                                    -type: 'AUTO', scalar or 
                                           numpy array [1, input_dim]

            initial_visible_offsets: Initial visible mean values.
                                    -type: 'AUTO', scalar or 
                                           numpy array [1, input dim]
                                  
            initial_hidden_offsets:  Initial hidden mean values.
                                    -type: 'AUTO', scalar or 
                                           numpy array [1, output_dim]
                                                                    
            dtype:                   Used data type.
                                    -type: numpy.float32, numpy.float64 and, 
                                           numpy.float128 
        '''
        # Call constructor of superclass
        super(GaussianBinaryVarianceRBM, 
              self).__init__(number_visibles = number_visibles, 
                             number_hiddens = number_hiddens, 
                             data = data,
                             initial_weights = initial_weights,
                             initial_visible_bias = initial_visible_bias,
                             initial_hidden_bias = initial_hidden_bias,
                             initial_sigma = initial_sigma,
                             initial_visible_offsets = initial_visible_offsets,
                             initial_hidden_offsets = initial_hidden_offsets, 
                             dtype = dtype)

    def _calculate_sigma_gradient(self, v, h):
        ''' This function calculates the gradient for the variance of the RBM.
        
        :Parameters:
            v: States of the visible variables.
                           -type: numpy arrays [batchsize, input dim]
                                
            h:   Probabilities of the hidden variables.
                           -type: numpy arrays [batchsize, output dim]
        
        :Returns:
            Sigma gradient.
           -type: list of numpy arrays [input dim,1]
        
        ''' 
        var_diff = (v - self.bv - self.ov)**2
        return ((var_diff - 2.0 * (v - self.ov) * numx.dot(h, 
                self.w.T)).sum(axis=0) / (self.sigma*self.sigma*self.sigma))

    def get_parameters(self):
        ''' This function returns all mordel parameters in a list.
        
        :Returns: 
            The parameter references in a list.
           -type: list 

        ''' 
        return [self.w, self.bv, self.bh, self.sigma]

    def calculate_gradients(self, v, h):
        ''' This function calculates all gradients of this RBM and returns 
            them as an ordered array. This keeps the flexibility of adding 
            parameters which will be updated by the training algorithms.
        
        :Parameters:
            v:   States of the visible variables.
                -type: numpy arrays [batchsize, input dim]
                                
            h:   Probabilities of the hidden variables.
                -type: numpy arrays [batchsize, output dim]
                                
        :Returns:
            Gradients for all parameters.
           -type: numpy arrays (num parameters x [parameter.shape])

        '''
        return [self._calculate_weight_gradient(v, h)
                ,self._calculate_visible_bias_gradient(v)
                ,self._calculate_hidden_bias_gradient(h)
                ,self._calculate_sigma_gradient(v, h)]
