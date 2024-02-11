import numpy as numx
from scipy.signal import convolve2d
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid
from sklearn.linear_model import LogisticRegression

class Unit_layer_binary_1D(object):
    
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

        if data != None:
            if isinstance(data,list):
                data = numx.concatenate(data)
            if self.input_dim != data.shape[1]:
                raise ValueError("Data dimension and model input \
                                     dimension have to be equal!")  
            data_mean = data.mean(axis=0).reshape(1,data.shape[1])
            
        # AUTO   -> data != None -> Initialized to the data mean
        #           data == None -> Initialized to Visible range mean
        # Scalar -> Initialized to given value
        # Array  -> The corresponding values are used  
        self.offset = numx.zeros((1,self.input_dim))
        if initial_offsets is 'AUTO':
            if data != None:
                self.offset += data_mean
            else:
                self.offset += 0.5
        else:
            if(numx.isscalar(initial_offsets)):
                self.offset += initial_offsets
            else:
                self.offset += initial_offsets.reshape(1,self.input_dim)
        self.offset = numx.array(self.offset, dtype=dtype)
        
        # AUTO   -> data != None -> Initialized to the inverse sigmoid of 
        #           data mean
        #           data == Initialized to randn()*0.01
        # Scalar -> Initialized to given value + randn()*0.01
        # Array  -> The corresponding values are used 
        self.bias = numx.zeros((1, self.input_dim))
        if initial_bias is 'AUTO':
            if data != None:
                self.bias = numx.array(Sigmoid.g(numx.clip(data_mean,0.001,
                          0.9999)), dtype=dtype).reshape(self.offset.shape)
        else:
            if initial_bias is 'INVERSE_SIGMOID':
                self.bias = numx.array(Sigmoid.g(numx.clip(self.offset,0.001,
                          0.9999)), dtype=dtype).reshape(1,self.input_dim)
            else:
                if(numx.isscalar(initial_bias)):
                    self.bias = numx.array(initial_bias 
                                     + numx.zeros((1, self.input_dim))
                                     , dtype=dtype) 
                else:
                    self.bias = numx.array(initial_bias, 
                                         dtype=dtype)

    def activation(self, bottom_up_states, top_down_states):
        ''' Calculates the pre and post synaptic activation.
            
        :Parameters:
            bottom_up_states: activation comming from previous layer.
                             -type: numpy array [batch_size, input dim]
                     
            top_down_states:  activation comming from next layer.
                             -type: numpy array [batch_size, input dim]

        '''
        pre_act = self.bias
        if self.input_weight_layer != None:
            pre_act = self.input_weight_layer.propagate_up(bottom_up_states) + pre_act
        if self.output_weight_layer != None:
            pre_act =self.output_weight_layer.propagate_down(top_down_states) + pre_act
        return Sigmoid.f(pre_act), pre_act

    def sample(self, activation):
        ''' This function samples states from the activation.
            
        :Parameters:
            activation: pre and post synaptiv activation.
                       -type: list len(2) of numpy arrays [batch_size, input dim]

        '''
        return self.dtype(activation[0] > numx.random.random(activation[0].shape))

    def calculate_probabilistic_gradient(self, 
                                         states_pos, 
                                         states_neg):
        ''' This function calculates the average gradient for the given data.
            
        :Parameters:
            x:        input data.
                     -type: numpy array [batch_size, input dim]

        '''
        return numx.mean(states_pos - states_neg, axis = 0).reshape(1,grad.shape[0])
    
    def update_biases(self, bias_updates, restriction, restriction_typ):
        ''' This function updates the bias parameter.
            
        :Parameters:
            bias_updates: Update for the bias parameter.
                      -type: numpy array [1, input dim]

        '''
        # Restricts the gradient
        if numx.isscalar(restriction):
            if restriction > 0:
                if restriction_typ is 'Cols' or restriction_typ is 'Rows' or restriction_typ is 'Mat':
                    bias_updates = numxExt.restrict_matrix_norm(
                                                    bias_updates,
                                                    restriction )
                    
                if restriction_typ is 'Val':
                    numx.clip(bias_updates, -restriction, restriction,bias_updates)
        self.bias += bias_updates
        
    def update_offsets(self, 
                       new_offset, 
                       offset_update,
                       bottom_up_offsets, 
                       top_down_offsets, 
                       bottom_up_weight_gradient, 
                       top_down_weight_gradient):
        ''' This function updates the weight parameter.
            
        :Parameters:
            new_offset:  Update for the bias parameter.
                            -type: numpy array [1, input dim]
                           
            offset_update: Update for the bias parameter.
                            -type: numpy array [1, input dim]

        '''
        if top_down_weight_gradient != None:
            self.bias += offset_update*numx.dot(top_down_offsets, top_down_weight_gradient.T)
        if bottom_up_weight_gradient != None:    
            grad -= numx.dot(bottom_up_offsets, bottom_up_weight_gradient)
        self.offset = (1.0-offset_update)*self.offset + offset_update*new_offset