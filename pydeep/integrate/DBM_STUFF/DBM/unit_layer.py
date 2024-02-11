import numpy as numx
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid

class Binary_layer(object):
    
    def __init__(self, 
                  input_weight_layer, 
                  output_weight_layer, 
                  data = None, 
                  initial_bias = 'AUTO', 
                  initial_offsets = 'AUTO',
                  dtype = numx.float64):
        ''' This function initializes the weight layer.
            
        :Parameters:
            input_weight_layer:      Referenz to the input weights.
                                    -type: Weight_layer or None
                                  
            output_weight_layer      Referenz to the output weights.
                                    -type: Weight_layer or None
                                        
            data:                    The training data for initializing the 
                                     visible bias.
                                    -type: None or 
                                           numpy array [num samples, input dim]
                                           or List of numpy arrays
                                           [num samples, input dim]
                                  
            initial_bias:            Initial visible bias.
                                    -type: 'AUTO', scalar or 
                                           numpy array [1,input dim]

            initial_offsets:         Initial visible mean values.
                                    -type: 'AUTO', scalar or 
                                           numpy array [1, input dim]
                                  
            dtype:                   Used data type i.e. numpy.float64
                                    -type: numpy.float32 or numpy.float64 or 
                                           numpy.float128  
            
        '''      
        # Set internal datatype
        self.dtype = dtype

        # Set input output dimesnions
        self.input_weight_layer = input_weight_layer
        self.output_weight_layer = output_weight_layer
        
        # Check that input and output layer match, which has not always to be the case e.g. SoftmaxUnitLayer
        if input_weight_layer != None: 
            self.input_dim = input_weight_layer.output_dim
            self.output_dim = self.input_dim
        else:
            if output_weight_layer != None:
                self.output_dim = output_weight_layer.input_dim
                self.input_dim = self.output_dim
            else:
                raise NotImplementedError("Unit layer needs at least one connected weight layer!")

        if data is not None:
            if isinstance(data,list):
                data = numx.concatenate(data)
            if self.input_dim != data.shape[1]:
                raise ValueError("Data dimension and model input \
                                     dimension have to be equal!")  
            data_mean = numx.mean(data,axis=0).reshape(1,data.shape[1])
            
        # AUTO   -> data is not None -> Initialized to the data mean
        #           data is None -> Initialized to Visible range mean
        # Scalar -> Initialized to given value
        # Array  -> The corresponding values are used  
        self.offset = numx.zeros((1,self.input_dim))
        if initial_offsets is 'AUTO':
            if data is not None:
                self.offset += data_mean
            else:
                self.offset += 0.5
        else:
            if(numx.isscalar(initial_offsets)):
                self.offset += initial_offsets
            else:
                self.offset += initial_offsets.reshape(1,self.input_dim)
        self.offset = numx.array(self.offset, dtype=dtype)
        
        # AUTO   -> data is not None -> Initialized to the inverse sigmoid of 
        #           data mean
        #           data is Initialized to randn()*0.01
        # Scalar -> Initialized to given value + randn()*0.01
        # Array  -> The corresponding values are used 
        self.bias = numx.zeros((1, self.input_dim))
        if initial_bias is 'AUTO':
            if data is not None:
                self.bias = numx.array(Sigmoid.g(numx.clip(data_mean,0.001,
                          0.999)), dtype=dtype).reshape(self.offset.shape)
        else:
            if initial_bias is 'INVERSE_SIGMOID':
                self.bias = numx.array(Sigmoid.g(numx.clip(self.offset,0.001,
                          0.999)), dtype=dtype).reshape(1,self.input_dim)
            else:
                if(numx.isscalar(initial_bias)):
                    self.bias = numx.array(initial_bias 
                                     + numx.zeros((1, self.input_dim))
                                     , dtype=dtype) 
                else:
                    self.bias = numx.array(initial_bias, 
                                         dtype=dtype)

    def activation(self, bottom_up_states, top_down_states, bottom_up_pre = None, top_down_pre = None):
        ''' Calculates the pre and post synaptic activation.
            
        :Parameters:
            bottom_up_states: activation comming from previous layer.
                             -type: numpy array [batch_size, input dim]
                     
            top_down_states:  activation comming from next layer.
                             -type: numpy array [batch_size, input dim]
                             
            bottom_up_pre:    pre-activation comming from previous layer of None.
                              if given this pre activation is used to avoid re-caluclations.
                             -type: None or numpy array [batch_size, input dim]
                     
            top_down_pre:     pre-activation comming from next layer of None.
                              if given this pre activation is used to avoid re-caluclations.
                             -type: None or numpy array [batch_size, input dim]
                             
        :Returns:
            Pre and post synaptic activation for this layer.
           -type: numpy array [batch_size, input dim]

        '''
        pre_act = 0.0
        if self.input_weight_layer is not None:
            if bottom_up_pre is None:
                pre_act += self.input_weight_layer.propagate_up(bottom_up_states)
            else:
                pre_act += bottom_up_pre
        if self.output_weight_layer is not None:
            if top_down_pre is None:
                pre_act +=self.output_weight_layer.propagate_down(top_down_states)
            else:
                pre_act += top_down_pre
        pre_act += self.bias
        return Sigmoid.f(pre_act), pre_act

    def sample(self, activation):
        ''' This function samples states from the activation.
            
        :Parameters:
            activation: pre and post synaptiv activation.
                       -type: list len(2) of numpy arrays [batch_size, input dim]

        '''
        return self.dtype(activation[0] > numx.random.random(activation[0].shape))

    def calculate_gradient_b(self, 
                    states_pos, 
                    states_neg, 
                    bottom_up_offsets, 
                    top_down_offsets, 
                    bottom_up_weight_gradient, 
                    top_down_weight_gradient):
        ''' This function calculates the average gradient for the given data.
            
        :Parameters:
            x:        input data.
                     -type: numpy array [batch_size, input dim]

        '''
        grad = numx.mean(states_pos - states_neg, axis = 0).reshape(1,self.input_dim)
        if bottom_up_weight_gradient is not None:    
            grad -= numx.dot(bottom_up_offsets, bottom_up_weight_gradient)
        if top_down_weight_gradient is not None:
            grad -= numx.dot(top_down_offsets, top_down_weight_gradient.T)
        return grad
    
    def update_biases(self, bias_updates, restriction , restriction_typ):
        ''' This function updates the weight parameter.
            
        :Parameters:
            b_updates: Update for the bias parameter.
                      -type: numpy array [1, input dim]

        '''
        # Restricts the gradient
        if numx.isscalar(restriction):
            if restriction > 0:
                if restriction_typ is 'Cols' or restriction_typ is 'Rows' or restriction_typ is 'Mat':
                    bias_updates = numxExt.restrict_norms(
                                                    bias_updates,
                                                    restriction )
                    
                if restriction_typ is 'Val':
                    numx.clip(bias_updates, -restriction, restriction,bias_updates)
        self.bias += bias_updates
        
    def update_offsets(self, offset_updates, shifting_factor):
        ''' This function updates the weight parameter.
            
        :Parameters:
            b_updates: Update for the bias parameter.
                      -type: numpy array [1, input dim]

        '''
        self.offset = (1.0-shifting_factor)*self.offset + shifting_factor*offset_updates
        
class Softmax_layer(Binary_layer):
    
    def __init__(self, 
                  input_weight_layer, 
                  output_weight_layer, 
                  data = None, 
                  initial_bias = 'AUTO', 
                  initial_offsets = 'AUTO',
                  dtype = numx.float64):
        ''' This function initializes the weight layer.
            
        :Parameters:
            input_weight_layer:      Referenz to the input weights.
                                    -type: Weight_layer or None
                                  
            output_weight_layer      Referenz to the output weights.
                                    -type: Weight_layer or None
                                        
            data:                    The training data for initializing the 
                                     visible bias.
                                    -type: None or 
                                           numpy array [num samples, input dim]
                                           or List of numpy arrays
                                           [num samples, input dim]
                                  
            initial_bias:            Initial visible bias.
                                    -type: 'AUTO', scalar or 
                                           numpy array [1,input dim]

            initial_offsets:         Initial visible mean values.
                                    -type: 'AUTO', scalar or 
                                           numpy array [1, input dim]
                                  
            dtype:                   Used data type i.e. numpy.float64
                                    -type: numpy.float32 or numpy.float64 or 
                                           numpy.float128  
            
        '''      
        # Call constructor of superclass
        super(Softmax_layer, 
              self).__init__(input_weight_layer, 
                             output_weight_layer, 
                             data, 
                             initial_bias, 
                             initial_offsets,
                             dtype)

    def activation(self, bottom_up_states, top_down_states, bottom_up_pre = None, top_down_pre = None):
        ''' Calculates the pre and post synaptic activation.
            
        :Parameters:
            bottom_up_states: activation comming from previous layer.
                             -type: numpy array [batch_size, input dim]
                     
            top_down_states:  activation comming from next layer.
                             -type: numpy array [batch_size, input dim]
                             
            bottom_up_pre:    pre-activation comming from previous layer of None.
                              if given this pre activation is used to avoid re-caluclations.
                             -type: None or numpy array [batch_size, input dim]
                     
            top_down_pre:     pre-activation comming from next layer of None.
                              if given this pre activation is used to avoid re-caluclations.
                             -type: None or numpy array [batch_size, input dim]
                             
        :Returns:
            Pre and post synaptic activation for this layer.
           -type: numpy array [batch_size, input dim]

        '''
        pre_act = self.bias
        if self.input_weight_layer is not None:
            if bottom_up_pre is None:
                pre_act = self.input_weight_layer.propagate_up(bottom_up_states) + pre_act
            else:
                pre_act = bottom_up_pre + pre_act
        if self.output_weight_layer is not None:
            if top_down_pre is None:
                pre_act =self.output_weight_layer.propagate_down(top_down_states) + pre_act
            else:
                pre_act = top_down_pre + pre_act
        return numx.exp(pre_act-numxExt.log_sum_exp(pre_act, axis = 1).reshape(pre_act.shape[0],1)), pre_act

    def sample(self, activation):
        ''' This function samples states from the activation.
            
        :Parameters:
            activation: pre and post synaptiv activation.
                       -type: list len(2) of numpy arrays [batch_size, input dim]

        '''
        probs = activation[0]/numx.sum(activation[0],axis = 1).reshape(activation[0].shape[0],1)
        mini = probs.cumsum(axis = 1)
        maxi = mini-probs
        sample = numx.random.random((activation[0].shape[0],1))
        return self.dtype((mini > sample)*(sample >= maxi))

class Gaussian_layer(Binary_layer):
    
    def __init__(self, 
                 input_weight_layer, 
                 output_weight_layer, 
                 data = None, 
                 initial_bias = 'AUTO', 
                 initial_offsets = 'AUTO',
                 dtype = numx.float64):
        ''' This function initializes the weight layer.
            
        :Parameters:
            input_weight_layer:      Referenz to the input weights.
                                    -type: Weight_layer or None
                                  
            output_weight_layer      Referenz to the output weights.
                                    -type: Weight_layer or None
                                        
            data:                    The training data for initializing the 
                                     visible bias.
                                    -type: None or 
                                           numpy array [num samples, input dim]
                                           or List of numpy arrays
                                           [num samples, input dim]
                                  
            initial_bias:            Initial visible bias.
                                    -type: 'AUTO', scalar or 
                                           numpy array [1,input dim]

            initial_offsets:         Initial visible mean values.
                                    -type: 'AUTO', scalar or 
                                           numpy array [1, input dim]
                                  
            dtype:                   Used data type i.e. numpy.float64
                                    -type: numpy.float32 or numpy.float64 or 
                                           numpy.float128  
            
        '''      
                # Call constructor of superclass
        super(Gaussian_layer, 
              self).__init__(input_weight_layer, 
                             output_weight_layer, 
                             data, 
                             initial_bias, 
                             initial_offsets,
                             dtype)

    def activation(self, bottom_up_states, top_down_states, bottom_up_pre = None, top_down_pre = None):
        ''' Calculates the pre and post synaptic activation.
            
        :Parameters:
            bottom_up_states: activation comming from previous layer.
                             -type: numpy array [batch_size, input dim]
                     
            top_down_states:  activation comming from next layer.
                             -type: numpy array [batch_size, input dim]

        '''
        pre_act = self.bias + self.offset
        if self.input_weight_layer is not None:
            pre_act = self.input_weight_layer.propagate_up(bottom_up_states) + pre_act
        if self.output_weight_layer is not None:
            pre_act =self.output_weight_layer.propagate_down(top_down_states) + pre_act
        return pre_act, pre_act

    def sample(self, activation):
        ''' This function samples states from the activation.
            
        :Parameters:
            activation: pre and post synaptiv activation.
                       -type: list len(2) of numpy arrays [batch_size, input dim]

        '''
        return activation[0] + numx.random.randn(activation[0].shape[0], activation[0].shape[1])
