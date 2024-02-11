import numpy as numx
from scipy.signal import convolve2d

import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid


class Weight_layer(object):
    ''' This class implements a weight layer that connects one unit 
        layer to another.
    
    '''

    @classmethod
    def generate_2D_connection_matrix(cls,
                                          input_x_dim, 
                                          input_y_dim, 
                                          field_x_dim, 
                                          field_y_dim, 
                                          overlap_x_dim, 
                                          overlap_y_dim, 
                                          wrap_around = True):
        ''' This function constructs a connection matrix, which can be 
            used to force the weights to have local receptive fields.
            
        :Parameters:
            input_x_dim:    Input dimension.
                           -type: int
                                  
            input_y_dim     Output dimension.
                           -type: int
                                        
            field_x_dim:    Size of the receptive field in dimension x.
                           -type: int

            field_y_dim:    Size of the receptive field in dimension y.
                           -type: int
                               
            overlap_x_dim:  Overlap of the receptive fields in dimension x.
                           -type: int

            overlap_y_dim:  Overlap of the receptive fields in dimension y.
                           -type: int
                           
            wrap_around:    If true teh overlap has warp around in both dimensions.
                           -type: bool

        :Returns:
            Connection matrix.
           -type: numpy arrays [input dim, output dim]
           
        '''   
        if field_x_dim > input_x_dim:
            raise NotImplementedError("field_x_dim > input_x_dim is invalid!")
        if field_y_dim > input_y_dim:
            raise NotImplementedError("field_y_dim > input_y_dim is invalid!")
        if overlap_x_dim >= field_x_dim:
            raise NotImplementedError("overlap_x_dim >= field_x_dim is invalid!")
        if overlap_y_dim >= field_y_dim:
            raise NotImplementedError("overlap_y_dim >= field_y_dim is invalid!")
        
        matrix = None
        start_x = 0
        start_y = 0
        end_x = input_x_dim 
        end_y = input_y_dim 
        if wrap_around == False:
            end_x -= field_x_dim - 1
            end_y -= field_y_dim - 1
        step_x = field_x_dim - overlap_x_dim
        step_y = field_y_dim - overlap_y_dim
        
        for x in range(start_x,end_x,step_x):
            for y in range(start_y,end_y,step_y):
                column = numx.zeros((input_x_dim,input_y_dim))
                for i in range(x,x+field_x_dim,1):
                    for j in range(y,y+field_y_dim,1):
                        column[i%input_x_dim,j%input_y_dim] = 1.0
                column = column.reshape((input_x_dim*input_y_dim))
                if matrix == None:
                    matrix = column
                else:
                    matrix = numx.vstack((matrix,column))
        return matrix.T

    def __init__(self, 
                  input_dim, 
                  output_dim, 
                  initial_weights = 'AUTO',
                  connections = None, 
                  dtype = numx.float64):
        ''' This function initializes the weight layer.
            
        :Parameters:
            input_dim:          Input dimension.
                               -type: int
                                  
            output_dim          Output dimension.
                               -type: int
                                        
            initial_weights:    Initial weights.
                                'AUTO' and a scalar are random init.
                               -type: 'AUTO', scalar or 
                                      numpy array [input dim, output_dim]

            connections:        Connection matrix containing 0 and 1 entries, where
                                0 connections disable the corresponding weight.
                                generate_2D_connection_matrix() can be used to contruct a matrix.
                               -type: numpy array [input dim, output_dim] or None
                               
            dtype:               Used data type i.e. numpy.float64
                                -type: numpy.float32 or numpy.float64 or 
                                       numpy.float128  
            
        '''      
        # Set internal datatype
        self.dtype = dtype

        # Set input and output dimension
        self.input_dim = input_dim
        self.output_dim = output_dim

        # AUTO   -> Small random values out of 
        #           +-4*numx.sqrt(6/(self.input_dim+self.output_dim)  
        # Scalar -> Small Gaussian distributed random values with std_dev 
        #           initial_weights
        # Array  -> The corresponding values are used   
        if initial_weights is 'AUTO':
            self.weights = numx.array((2.0 * numx.random.rand(self.input_dim,
                                                self.output_dim) - 1.0)
                              * (4.0 * numx.sqrt(6.0 / (self.input_dim 
                                                    + self.output_dim)))
                              ,dtype=dtype)
        else:
            if(numx.isscalar(initial_weights)):
                self.weights = numx.array(numx.random.randn(self.input_dim, 
                                                      self.output_dim) 
                                  * initial_weights, dtype=dtype)
            else:
                if initial_weights.shape[0] != self.input_dim or initial_weights.shape[1] != self.output_dim or len(initial_weights.shape) != 2:
                    raise NotImplementedError("Initial weight matrix must habe the shape defined by input_dim and output_dim!")
                else:
                    self.weights = numx.array(initial_weights, dtype=dtype)
                    
        # Set connection matrix
        self.connections = None
        if connections != None:
            if connections.shape[0] != self.input_dim or connections.shape[1] != self.output_dim or len(connections.shape) != 2:
                raise NotImplementedError("Connections matrix must have the shape defined by input_dim and output_dim!")
            else:
                self.connections = numx.array(connections, dtype=dtype)
                self.weights *= self.connections

    def propagate_up(self, bottom_up_states):
        ''' This function propagates the input to the next layer.
            
        :Parameters:
            bottom_up_states: States of the unit layer below.
                             -type: numpy array [batch_size, input dim]

        :Returns:
            Top down states.
           -type: numpy arrays [batch_size, output dim]

        '''
        return numx.dot(bottom_up_states,self.weights)
    
    def propagate_down(self, top_down_states):
        ''' This function propagates the output to the previous layer.
            
        :Parameters:
            top_down_states: States of the unit layer above.
                            -type: numpy array [batch_size, output dim]

        :Returns:
            Bottom up states.
           -type: numpy arrays [batch_size, input dim]

        '''
        return numx.dot(top_down_states,self.weights.T)
        
    def calculate_weight_gradients(self, 
                                      bottom_up_pos, 
                                      top_down_pos, 
                                      bottom_up_neg, 
                                      top_down_neg, 
                                      offset_bottom_up, 
                                      offset_top_down):
        ''' This function calculates the average gradient from input 
            and output samples of positive and negative sampling phase.
            
        :Parameters:
            bottom_up_pos:    Input samples from the positive phase.
                             -type: numpy array [batch_size, input dim]

            top_down_pos:     Output samples from the positive phase.
                             -type: numpy array [batch_size, output dim]
                 
            bottom_up_neg:    Input samples from the negative phase.
                             -type: numpy array [batch_size, input dim]

            top_down_neg:     Output samples from the negative phase.
                             -type: numpy array [batch_size, output dim]
                      
            offset_bottom_up: Offset for the input data.
                             -type: numpy array [1, input dim]
                 
            offset_top_down:  Offset for the output data.
                             -type: numpy array [1, output dim]
                            
        :Returns:
            Weight gradient.
           -type: numpy arrays [input dim, output dim]

        '''
        pos = numx.dot((bottom_up_pos-offset_bottom_up).T,(top_down_pos-offset_top_down))
        neg = numx.dot((bottom_up_neg-offset_bottom_up).T,(top_down_neg-offset_top_down))
        grad = (pos-neg)/bottom_up_pos.shape[0]
        if self.connections != None:
            grad *= self.connections
        return grad
    
    def update_weights(self, weight_updates, restriction, restriction_typ):
        ''' This function updates the weight parameters.
            
        :Parameters:
            weight_updates:     Update for the weight parameter.
                               -type: numpy array [input dim, output dim]
                              
            restriction:        If a scalar is given the weights will be 
                                forced after an update not to exceed this value. 
                                restriction_typ controls how the values are 
                                restricted.
                               -type: scalar or None
                                      
            restriction_typ:    If a value for the restriction is given, this parameter
                                determines the restriction typ. 'Cols', 'Rows', 'Mat' 
                                or 'Abs' to restricted the colums, rows or matrix norm 
                                or the matrix absolut values.
                               -type: string

        '''
        
        # Restricts the gradient
        if numx.isscalar(restriction):
            if restriction > 0:
                if restriction_typ is 'Cols':
                    weight_updates = numxExt.restrict_norms(
                                                    weight_updates,
                                                    restriction, 0 )
                if restriction_typ is 'Rows':
                    weight_updates = numxExt.restrict_norms(
                                                    weight_updates,
                                                    restriction, 1 )
                if restriction_typ is 'Mat':
                    weight_updates = numxExt.restrict_matrix_norm(
                                                    weight_updates,
                                                    restriction )
                    
                if restriction_typ is 'Val':
                    numx.clip(weight_updates, -restriction, restriction,weight_updates)
                    
        # Update weights
        self.weights += weight_updates
             
class Convolutional_weight_layer(Weight_layer):
    ''' This class implements a weight layer that connects one unit 
        layer to another with convolutional weights.
        
    '''

    @classmethod
    def construct_gauss_filter(cls, width, height, variance = 1.0):
        ''' This function constructs a 2D-Gauss filter.
        
        :Parameters:
            width:     Filter width.
                      -type: int
                                  
            height     Filter Height.
                      -type: int
                   
            variance   Variance of the Gaussian
                      -type: scalar
                                        

        :Returns:
            Convolved matrix with the same shape as matrix.
           -type: 2D numpy arrays
           
        ''' 
        if width % 2 == 0:
            print "Width needs to be odd!"
            pass
        if height % 2 == 0:
            print "Height needs to be odd!"
            pass
        lowerW = (width-1)/2
        lowerH = (height-1)/2
        mat = numx.zeros((width,height))
        for x in range(0,width):
            for y in range(0,height):
                mat[x,y] = (numx.exp(-0.5*((x-lowerW)**2+(y-lowerH)**2)/variance))/(2*numx.pi*variance)
        return mat/numx.sum(mat)

    def __convolve(self, matrix, mask):
        ''' This function performs a 2D convolution on every column of a given matrix.
        
        :Parameters:
            matrix: 2D-input matrix.
                   -type: small 2D numpy array
                                  
            mask    2D-mask matrix.
                   -type: small 2D numpy array
                                        

        :Returns:
            Convolved matrix with the same shape as matrix.
           -type: 2D numpy arrays
           
        '''   
        s = numx.int32(numx.sqrt(matrix.shape[1]))
        result = numx.empty(matrix.shape)
        for i in range(matrix.shape[0]):
            temp = matrix[i].reshape(s,s) 
            result[i] = convolve2d(temp, mask, mode='same', boundary='wrap').reshape(matrix.shape[1])
        return result 

    def __init__(self, 
                  input_dim, 
                  output_dim, 
                  mask,
                  initial_weights = 'AUTO',
                  connections = None, 
                  dtype = numx.float64):
        ''' This function initializes the convolutional weight layer.
            
        :Parameters:
            input_dim:          Input dimension.
                               -type: int
                                  
            output_dim          Output dimension.
                               -type: int
                               
            mask                Convolution mask.
                                construct_gauss_filter can be used for example.
                               -type: small numpy array
   
            initial_weights:    Initial weights.
                                'AUTO' and a scalar are random init.
                               -type: 'AUTO', scalar or 
                                      numpy array [input dim, output_dim]

            connections:        Connection matrix containing 0 and 1 entries, where
                                0 connections disable the corresponding weight.
                                generate_2D_connection_matrix() can be used to contruct a matrix.
                               -type: numpy array [input dim, output_dim] or None
                               
            dtype:               Used data type i.e. numpy.float64
                                -type: numpy.float32 or numpy.float64 or 
                                       numpy.float128  
            
        '''    
        super(Convolutional_weight_layer, 
              self).__init__(input_dim = input_dim, 
                             output_dim = output_dim, 
                             initial_weights = initial_weights,
                             connections = connections, 
                             dtype = dtype)
        self.mask = mask     
        self.orginalW = numx.copy(self.weights)      
        self.weights = self.__convolve(self.orginalW,self.mask)

    def calculate_weight_gradients(self, 
                                      bottom_up_pos, 
                                      top_down_pos, 
                                      bottom_up_neg, 
                                      top_down_neg, 
                                      offset_bottom_up, 
                                      offset_top_down):
        ''' This function calculates the average gradient from input 
            and output samples of positive and negative sampling phase.
            
        :Parameters:
            bottom_up_pos:    Input samples from the positive phase.
                             -type: numpy array [batch_size, input dim]

            top_down_pos:     Output samples from the positive phase.
                             -type: numpy array [batch_size, output dim]
                 
            bottom_up_neg:    Input samples from the negative phase.
                             -type: numpy array [batch_size, input dim]

            top_down_neg:     Output samples from the negative phase.
                             -type: numpy array [batch_size, output dim]
                      
            offset_bottom_up: Offset for the input data.
                             -type: numpy array [1, input dim]
                 
            offset_top_down:  Offset for the output data.
                             -type: numpy array [1, output dim]
                            
        :Returns:
            Weight gradient.
           -type: numpy arrays [input dim, output dim]

        '''
        pos = numx.dot((bottom_up_pos-offset_bottom_up).T,(self.__convolve(top_down_pos-offset_top_down,self.mask)))
        neg = numx.dot((bottom_up_neg-offset_bottom_up).T,(self.__convolve(top_down_neg-offset_top_down,self.mask)))
        grad = (pos-neg)/bottom_up_pos.shape[0]
        if self.connections != None:
            grad *= self.connections
        return grad
    
    def update_weights(self, weight_updates, restriction, restriction_typ):
        ''' This function updates the weight parameters.
            
        :Parameters:
            weight_updates:     Update for the weight parameter.
                               -type: numpy array [input dim, output dim]
                              
            restriction:        If a scalar is given the weights will be 
                                forced after an update not to exceed this value. 
                                restriction_typ controls how the values are 
                                restricted.
                               -type: scalar or None
                                      
            restriction_typ:    If a value for the restriction is given, this parameter
                                determines the restriction typ. 'Cols', 'Rows', 'Mat' 
                                or 'Abs' to restricted the colums, rows or matrix norm 
                                or the matrix absolut values.
                               -type: string
           
        '''
        # Update weights
        self.orginalW += weight_updates

        # Restricts the gradient
        if numx.isscalar(restriction):
            if restriction > 0:
                if restriction_typ is 'Cols':
                    self.orginalW = numxExt.restrict_norms(
                                                    self.orginalW,
                                                    restriction, 0 )
                if restriction_typ is 'Rows':
                    self.orginalW = numxExt.restrict_norms(
                                                    self.orginalW,
                                                    restriction, 1 )
                if restriction_typ is 'Mat':
                    self.orginalW = numxExt.restrict_matrix_norm(
                                                    self.orginalW,
                                                    restriction )
                    
                if restriction_typ is 'Val':
                    numx.clip(self.orginalW, -restriction, restriction,self.orginalW)
        
        self.weights = self.__convolve(self.orginalW,self.mask)

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
        if self.input_weight_layer != None:
            if bottom_up_pre == None:
                pre_act = self.input_weight_layer.propagate_up(bottom_up_states) + pre_act
            else:
                pre_act = bottom_up_pre + pre_act
        if self.output_weight_layer != None:
            if top_down_pre == None:
                pre_act =self.output_weight_layer.propagate_down(top_down_states) + pre_act
            else:
                pre_act = top_down_pre + pre_act
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
        grad = numx.mean(states_pos - states_neg, axis = 0)
        grad = grad.reshape(1,grad.shape[0])
        if top_down_weight_gradient != None:
            grad -= numx.dot(top_down_offsets, top_down_weight_gradient.T)
        if bottom_up_weight_gradient != None:    
            grad -= numx.dot(bottom_up_offsets, bottom_up_weight_gradient)
        return grad
    
    def update_biases(self, bias_updates, restriction, restriction_typ):
        ''' This function updates the weight parameter.
            
        :Parameters:
            b_updates: Update for the bias parameter.
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
        if self.input_weight_layer != None:
            if bottom_up_pre == None:
                pre_act = self.input_weight_layer.propagate_up(bottom_up_states) + pre_act
            else:
                pre_act = bottom_up_pre + pre_act
        if self.output_weight_layer != None:
            if top_down_pre == None:
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
        if self.input_weight_layer != None:
            pre_act = self.input_weight_layer.propagate_up(bottom_up_states) + pre_act
        if self.output_weight_layer != None:
            pre_act =self.output_weight_layer.propagate_down(top_down_states) + pre_act
        return pre_act, pre_act

    def sample(self, activation):
        ''' This function samples states from the activation.
            
        :Parameters:
            activation: pre and post synaptiv activation.
                       -type: list len(2) of numpy arrays [batch_size, input dim]

        '''
        return activation[0] + numx.random.randn(activation[0].shape[0], activation[0].shape[1])

"""
import numpy as numx
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.measuring as MEASURE
import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO

# Set the same seed value for all algorithms
numx.random.seed(42)


def int_to_bin(array):
    maxlabel = numx.max(array)+1
    result = numx.zeros((array.shape[0],maxlabel))
    for i in range(array.shape[0]):
        result[i,array[i]] = 1
    return result

v11 = v12 = 32
v21 = 25
v22 = 25
v31 = 25
v32 = 25

data1 , label1  = IO.load_CIFAR("../../../../../data/CIFAR/data_batch_1",True)
data2 , label2  = IO.load_CIFAR("../../../../../data/CIFAR/data_batch_2",True)
data3 , label3  = IO.load_CIFAR("../../../../../data/CIFAR/data_batch_3",True)
data4 , label4  = IO.load_CIFAR("../../../../../data/CIFAR/data_batch_4",True)
data5 , label5  = IO.load_CIFAR("../../../../../data/CIFAR/data_batch_5",True)
data = numx.vstack((data1,data2,data3,data4,data5))
label = numx.vstack((label1,label2,label3,label4,label5))
#data = numx.random.permutation(IO.load_matlab_file('../../../../workspacePy/data/NaturalImage.mat','rawImages'))

data = PRE.remove_rows_means(data)

zca = PRE.STANDARIZER(v11*v12)
zca.train_images(data)
data = zca.project(data)
max_norm = []
norm_typ = []
max_norm.append(0.01*numx.max(numxExt.get_norms(data, axis = 1)))
norm_typ.append('Cols')
max_norm.append(0.01)
norm_typ.append('Cols')
max_norm.append(0.01)
norm_typ.append('Cols')

data = [data,None,None]

N = v11 * v12
M = v21 * v22
O = v31 * v32
cons = Weight_layer.generate_2D_connection_matrix(v11, v12, 8, 8, 7, 7, False)

wl1 = Weight_layer(input_dim = N, 
                   output_dim = M, 
                   initial_weights = 0.01,
                   connections = cons,
                   dtype = numx.float64)
                      
wl2 = Weight_layer(input_dim = M, 
                   output_dim = O, 
                   initial_weights = 0.01,
                   connections = None,
                   dtype = numx.float64)

l1 =Gaussian_layer(None, 
                  wl1, 
                  data = data[0], 
                  initial_bias = 0,
                  initial_offsets = 0.0,
                  dtype = numx.float64)

l2 = Binary_layer(wl1, 
                  wl2, 
                  data = data[1], 
                  initial_bias = -4.0,
                  initial_offsets = 0.0,
                  dtype = numx.float64)

l3 = Binary_layer(wl2, 
                  None, 
                  data = data[2], 
                  initial_bias = -4.0,
                  initial_offsets = 0.0,
                  dtype = numx.float64)

layers = []
layers.append(l1)
layers.append(l2)
layers.append(l3)

model = MODEL.DBM_model(layers)

# Initialize parameters
max_epochs = 500
batch_size = 20
k_pos = 3
k_neg = 1

lr_W = numx.array([1.0,1.0])*0.01

lr_b =  numx.array([0.0,1.0,1.0])*0.01

lr_o =  numx.array([0.0,1.0,1.0])*0.01


'''
0.0486933539217     0.585254785998     0.179284586312     0.175034618955
0.0390717967691     0.824093347994     0.173895803368     0.160664040382
0.0334321437693     1.0067367206     0.175473179116     0.152322410028
0.0303970454395     1.15231618878     0.179143987883     0.147340164174
0.0278216872504     1.27830494277     0.184351540106     0.144885449775
0.0262050614064     1.38765220238     0.190983765155     0.144049116386
0.0246086691701     1.48518634491     0.198568146251     0.143241249658
0.0236067326334     1.57737898685     0.207775860588     0.142971335096
0.022594973556     1.65834878662     0.214352430841     0.143022302609
0.021672362472     1.73644843867     0.22287414777     0.142959954867

End-time:     2015-03-18 14:57:34.618567
Training time:    0:04:41.158776

'''

num_layers = len(layers)
mom = 0.9
# Initialize negative Markov chain
neg_chains = []
wl_grad = []
b_grad = []
for i in range(num_layers):
    neg_chains.append(numx.zeros((batch_size,layers[i].input_dim))+layers[i].offset)
    if i < num_layers-1:
        wl_grad.append(numx.zeros(layers[i].output_weight_layer.weights.shape))
    b_grad.append(numx.zeros(layers[i].bias.shape))


# Reparameterize RBM such that the inital setting is the same for centereing and centered training
layers[0].bias += numx.dot(0.0-layers[1].offset,layers[0].output_weight_layer.weights.T)
for i in range(1,num_layers-1):
    layers[i].bias += numx.dot(0.0-layers[i-1].offset,layers[i-1].output_weight_layer.weights) + numx.dot(0.0-layers[i+1].offset,layers[i+1].input_weight_layer.weights.T)
layers[num_layers-1].bias += numx.dot(0.0-layers[num_layers-2].offset,layers[num_layers-2].output_weight_layer.weights)

# Start time measure and training
measurer = MEASURE.Stopwatch()

if num_layers % 2 == 0:
    even_num_layers = True
else:
    even_num_layers = False
for epoch in range(0,max_epochs) :
    re = 0
    mom*=0.95
    for b in range(0,data[0].shape[0],batch_size):
        batch = []
        for i in range(len(data)):
            if data[i] != None:
                batch.append(data[i][b:b+batch_size,:])
            else:
                batch.append(None)

        pos_chains = model.meanfield_estimate(batch,k_pos,None)
        model.sample(neg_chains,k_neg,neg_chains)
        
        #pos_chains[1] = layers[1].sample([pos_chains[1]])
        #pos_chains = []
        #pos_chains.append(numx.copy(batch[0]))
        #pos_chains.append(layers[1].sample(layers[1].activation(pos_chains[0],None, None, None)))
        
        #neg_chains[0] = layers[0].sample(layers[0].activation(None,pos_chains[1], None, None))
        #neg_chains[1] = layers[1].sample(layers[1].activation(neg_chains[0],None, None, None))
        #model.sample(neg_chains,k_neg,neg_chains)

        # Estimate new means and update
        for i in range(num_layers):
            layers[i].update_offsets(numx.mean(pos_chains[i],axis = 0).reshape(1,pos_chains[i].shape[1]), lr_o[i])
        
        # Calculate centered weight gradients and update
        for i in range(num_layers-1):
            wl_grad[i] = wl_grad[i]*mom + layers[i].output_weight_layer.calculate_weight_gradients(pos_chains[i], pos_chains[i+1], neg_chains[i], neg_chains[i+1], layers[i].offset, layers[i+1].offset)
            layers[i].output_weight_layer.update_weights(lr_W[i]*wl_grad[i], max_norm[i], norm_typ[i])

        
        # Calsulate centered gradients for biases and update
        b_grad[0] = b_grad[0]*mom + layers[0].calculate_gradient_b(pos_chains[0], neg_chains[0], None, layers[1].offset, None, wl_grad[0])
        for i in range(1,num_layers-1):   
            b_grad[i] = b_grad[i]*mom + layers[i].calculate_gradient_b(pos_chains[i], neg_chains[i], layers[i-1].offset, layers[i+1].offset, wl_grad[i-1], wl_grad[i])
        b_grad[num_layers-1] = b_grad[num_layers-1]*mom + layers[num_layers-1].calculate_gradient_b(pos_chains[num_layers-1], neg_chains[num_layers-1], layers[num_layers-2].offset, None, wl_grad[num_layers-2], None)

        #Einbauen letzten zustande saven in Unit layer
        for i in range(num_layers):   
            layers[i].update_biases(lr_b[i]*b_grad[i], max_norm[i], norm_typ[i])
            
        re += numx.mean((layers[0].activation(None,pos_chains[1])[0]-pos_chains[0])**2)      
        
    # Plot Error and parameter norms, should be the same for both variants
    print re/data[0].shape[0]*batch_size,'\t',
    #for i in range(num_layers-1): 
    #    print numx.mean(numxExt.get_norms(layers[i].output_weight_layer.weights)),'\t',
    print " "
    
# End time measure
measurer.end()
print
print 'End-time: \t', measurer.get_end_time()
print 'Training time:\t', measurer.get_interval()
VIS.imshow_matrix(VIS.tile_matrix_rows(pos_chains[0][0:batch_size].T, v11,v12, batch_size/10,10, border_size = 1,normalized = False), 'data') 
VIS.imshow_matrix(VIS.tile_matrix_rows(layers[0].activation(None,pos_chains[1])[0][0:batch_size].T, v11,v12, batch_size/10,10, border_size = 1,normalized = False), 'sample 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(pos_chains[1][0:batch_size].T, v21,v22, batch_size/10,10, border_size = 1,normalized = False), 'sample 2')
VIS.imshow_matrix(VIS.tile_matrix_rows(neg_chains[0][0:batch_size].T, v11,v12, batch_size/10,10, border_size = 1,normalized = False), 'datan') 
VIS.imshow_matrix(VIS.tile_matrix_rows(layers[0].activation(None,neg_chains[1])[0][0:batch_size].T, v11,v12, batch_size/10,10, border_size = 1,normalized = False), 'sample 1n')
VIS.imshow_matrix(VIS.tile_matrix_rows(neg_chains[1][0:batch_size].T, v21,v22, batch_size/10,10, border_size = 1,normalized = False), 'sample 2n')

VIS.imshow_matrix(VIS.tile_matrix_rows(layers[0].output_weight_layer.weights, v11,v12, v21,v22, border_size = 1,normalized = False), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(layers[0].output_weight_layer.weights,layers[1].output_weight_layer.weights), v11,v12, v31,v32, border_size = 1,normalized = False), 'Weights 2')

VIS.show()


exit()























# Load Data
data , label , _ , _ , test , test_l = IO.load_MNIST("../../../../workspacePy/data/mnist.pkl.gz",False)
label = int_to_bin(label)
test_l = int_to_bin(test_l)
data = [data,None,None, label]#,None,label]
# Set dimensions
v11 = v12 = 28
v21 = 10
v22 = 10
v31 = 10
v32 = 10
v41 = 2
v42 = 5
v51 = 5
v52 = 2

N = v11 * v12
M = v21 * v22
O = v31 * v32
P = v41 * v42
Q = v51 * v52

connections = Weight_layer.generate_2D_connection_matrix(28, 28, 9, 9, 8, 8, False)
           
wl1 = Weight_layer(input_dim = N, 
                   output_dim = M, 
                   initial_weights = 0.01,
                   connections = None,#connections, 
                   dtype = numx.float64)
                          
wl2 = Weight_layer(input_dim = M, 
                                 #mask = Convolutional_weight_layer.construct_gauss_filter(3,3, variance = 0.5),
                                 output_dim = O, 
                                 initial_weights = 0.01,
                                 connections = None,#connections, 
                                 dtype = numx.float64)

wl3 = Weight_layer(input_dim = O, 
                   output_dim = P, 
                   initial_weights = 0.01,
                   connections = None,#Weight_layer.generate_2D_connection_matrix(14, 14, 8, 8, 7, 7, False), 
                   dtype = numx.float64)
'''     
wl4 = Weight_layer(input_dim = P, 
                   output_dim = P, 
                   initial_weights = 0.01,
                   connections = None,
                   dtype = numx.float64)
'''


l1 = Binary_layer(None, 
                  wl1, 
                  data = data[0], 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

l2 = Binary_layer(wl1, 
                  wl2, 
                  data = data[1], 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

l3 = Binary_layer(wl2, 
                  wl3, 
                  data = data[2], 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

l4 = Binary_layer(wl3, 
                   None, 
                   data = data[3], 
                   initial_bias = 0,
                   initial_offsets = 0,
                   dtype = numx.float64)
'''
l5 = Binary_layer(wl4, 
                  None, 
                  data = None, 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)
'''
layers = []
layers.append(l1)
layers.append(l2)
layers.append(l3)
layers.append(l4)
model = MODEL.DBM_model(layers)

#layers.append(l5)

# Initialize parameters
max_epochs = 25
batch_size = 100
k_pos = 5
k_neg = 3

lr_W = numx.array([0.01,0.01,0.01])

lr_b =  numx.array([0.01,0.01,0.01,0.01])

lr_o =  numx.array([0.01,0.01,0.01,0.0])


'''
0.0486933539217     0.585254785998     0.179284586312     0.175034618955
0.0390717967691     0.824093347994     0.173895803368     0.160664040382
0.0334321437693     1.0067367206     0.175473179116     0.152322410028
0.0303970454395     1.15231618878     0.179143987883     0.147340164174
0.0278216872504     1.27830494277     0.184351540106     0.144885449775
0.0262050614064     1.38765220238     0.190983765155     0.144049116386
0.0246086691701     1.48518634491     0.198568146251     0.143241249658
0.0236067326334     1.57737898685     0.207775860588     0.142971335096
0.022594973556     1.65834878662     0.214352430841     0.143022302609
0.021672362472     1.73644843867     0.22287414777     0.142959954867

End-time:     2015-03-18 14:57:34.618567
Training time:    0:04:41.158776

'''

num_layers = len(layers)
print num_layers
# Initialize negative Markov chain
neg_chains = []
for i in range(num_layers):
    neg_chains.append(numx.zeros((batch_size,layers[i].input_dim))+layers[i].offset)

# Reparameterize RBM such that the inital setting is the same for centereing and centered training
layers[0].bias += numx.dot(0.0-layers[1].offset,layers[0].output_weight_layer.weights.T)
for i in range(1,num_layers-1):
    layers[i].bias += numx.dot(0.0-layers[i-1].offset,layers[i-1].output_weight_layer.weights) + numx.dot(0.0-layers[i+1].offset,layers[i+1].input_weight_layer.weights.T)
layers[num_layers-1].bias += numx.dot(0.0-layers[num_layers-2].offset,layers[num_layers-2].output_weight_layer.weights)

# Start time measure and training
measurer = MEASURE.Stopwatch()

if num_layers % 2 == 0:
    even_num_layers = True
else:
    even_num_layers = False
for epoch in range(0,max_epochs) :
    re = 0
    for b in range(0,data[0].shape[0],batch_size):
        batch = []
        for i in range(len(data)):
            if data[i] != None:
                batch.append(data[i][b:b+batch_size,:])
            else:
                batch.append(None)
        '''
        #print b/numx.float64(data.shape[0])
        pre_top = []
        pre_bottom = []
        for i in range(num_layers):
            pre_top.append(None)
            pre_bottom.append(None)
        for i in range(num_layers):
            if data[i] != None:
                if i > 0:
                    pre_top[i-1] = layers[i].input_weight_layer.propagate_down(data[i][b:b+batch_size,:])
                if i < num_layers-1:
                    pre_bottom[i+1] = layers[i].output_weight_layer.propagate_up(data[i][b:b+batch_size,:])
    
        #positive phase
        pos_chains = []
        for i in range(0,num_layers):
            if data[i] != None:
                pos_chains.append(data[i][b:b+batch_size,:])
            else:
                pos_chains.append(numx.zeros((batch_size,layers[i].input_dim))+layers[i].offset)

        for i in range(k_pos):
            for i in range(1,num_layers-1,2):
                if data[i] == None:
                    pos_chains[i] = layers[i].activation(pos_chains[i-1], pos_chains[i+1], pre_bottom[i], pre_top[i])[0]

            #Top layer if even
            if even_num_layers == True:
                if data[num_layers-1] == None:
                    pos_chains[num_layers-1] = layers[num_layers-1].activation(pos_chains[num_layers-2], None, pre_bottom[num_layers-1], pre_top[num_layers-1])[0]

            # First Layer
            if data[0] == None:
                pos_chains[0] = layers[0].activation(None,pos_chains[1],pre_bottom[0], pre_top[0])[0]

            for i in range(2,num_layers-1,2):
                if data[i] == None:
                    pos_chains[i] = layers[i].activation(pos_chains[i-1], pos_chains[i+1], pre_bottom[i], pre_top[i])[0]

            #Top layer if odd
            if even_num_layers == False:
                if data[num_layers-1] == None:
                    pos_chains[num_layers-1] = layers[num_layers-1].activation(pos_chains[num_layers-2], None, pre_bottom[num_layers-1], pre_top[num_layers-1])[0]
        '''
        pos_chains = model.meanfield_estimate(batch,k_pos,None)
        model.sample(neg_chains,k_neg,neg_chains)
        '''
        #negative phase
        for i in range(k_neg):
            for i in range(1,num_layers-1,2):
                neg_chains[i] = layers[i].sample(layers[i].activation(neg_chains[i-1], neg_chains[i+1]))
            #Top layer if even
            if even_num_layers == True:
                neg_chains[num_layers-1] = layers[num_layers-1].sample(layers[num_layers-1].activation(neg_chains[num_layers-2], None))
            # First Layer
            neg_chains[0] = layers[0].sample(layers[0].activation(None,neg_chains[1]))
            for i in range(2,num_layers-1,2):
                neg_chains[i] = layers[i].sample(layers[i].activation(neg_chains[i-1], neg_chains[i+1]))
            #Top layer if odd
            if even_num_layers == False:
                neg_chains[num_layers-1] = layers[num_layers-1].sample(layers[num_layers-1].activation(neg_chains[num_layers-2], None))
        '''
        # Estimate new means and update
        for i in range(num_layers):
            layers[i].update_offsets(numx.mean(pos_chains[i],axis = 0).reshape(1,pos_chains[i].shape[1]), lr_o[i])
        
        # Calculate centered weight gradients and update
        wl_grad = []
        for i in range(num_layers-1):
            wl_grad.append(layers[i].output_weight_layer.calculate_weight_gradients(pos_chains[i], pos_chains[i+1], neg_chains[i], neg_chains[i+1], layers[i].offset, layers[i+1].offset))
            layers[i].output_weight_layer.update_weights(lr_W[i]*wl_grad[i], None, None)

        
        # Calsulate centered gradients for biases and update
        b_grad = []
        b_grad.append(layers[0].calculate_gradient_b(pos_chains[0], neg_chains[0], None, layers[1].offset, None, wl_grad[0]))
        for i in range(1,num_layers-1):   
            b_grad.append(layers[i].calculate_gradient_b(pos_chains[i], neg_chains[i], layers[i-1].offset, layers[i+1].offset, wl_grad[i-1], wl_grad[i]))
        b_grad.append(layers[num_layers-1].calculate_gradient_b(pos_chains[num_layers-1], neg_chains[num_layers-1], layers[num_layers-2].offset, None, wl_grad[num_layers-2], None))

        #Einbauen letzten zustande saven in Unit layer
        for i in range(num_layers):   
            layers[i].update_biases(lr_b[i]*b_grad[i])
            
        re += numx.mean((layers[0].activation(None,pos_chains[1])[0]-pos_chains[0])**2)
    '''
    # Estimate Meanfield reconstruction 
    pos_chains = []
    pos_chains.append(data)
    for i in range(1,num_layers):
        pos_chains.append(numx.zeros((data.shape[0],layers[i].input_dim))+layers[i].offset)

    for i in range(k_pos):
        for i in range(1,num_layers-1,2):
            pos_chains[i] = layers[i].activation(pos_chains[i-1], pos_chains[i+1])[0]
        #Top layer if even
        if even_num_layers == True:
            pos_chains[num_layers-1] = layers[num_layers-1].activation(pos_chains[num_layers-2], None)[0]
        # First Layer
        pos_chains[0] = layers[0].activation(None,pos_chains[1])[0]
        for i in range(2,num_layers-1,2):
            pos_chains[i] = layers[i].activation(pos_chains[i-1], pos_chains[i+1])[0]
        #Top layer if odd
        if even_num_layers == False:
            pos_chains[num_layers-1] = layers[num_layers-1].activation(pos_chains[num_layers-2], None)[0]
    '''
            
    # Plot Error and parameter norms, should be the same for both variants
    print re/data[0].shape[0]*batch_size,'\t',
    #for i in range(num_layers-1): 
    #    print numx.mean(numxExt.get_norms(layers[i].output_weight_layer.weights)),'\t',
    print " "
    #print numx.mean(numxExt.get_norms(layers[1].output_weight_layer.weights)),'\t',numx.mean(numxExt.get_norms(layers[0].bias)),'\t',
    #print numx.mean(numxExt.get_norms(l2.bias)),'\t', numx.mean(numxExt.get_norms(layers[2].bias)),'\t',
    #print numx.mean(layers[0].offset),'\t',numx.mean(l2.offset),'\t',numx.mean(layers[2].offset)

    act = model.meanfield_estimate([data[0],None,None, None],k_pos,None)[3]
    pred = maxOut = numx.int32(act >= numx.atleast_2d(numx.max(act,axis = 1)).T)
    print numx.mean(numx.sum(numx.abs(0.5*numx.abs(label-pred)),axis=1))
    
    act = model.meanfield_estimate([test,None,None, None],k_pos,None)[3]
    pred = maxOut = numx.int32(act >= numx.atleast_2d(numx.max(act,axis = 1)).T)
    print numx.mean(numx.sum(numx.abs(0.5*numx.abs(test_l-pred)),axis=1))
    
# End time measure
measurer.end()
print
print 'End-time: \t', measurer.get_end_time()
print 'Training time:\t', measurer.get_interval()
VIS.imshow_matrix(VIS.tile_matrix_rows(pos_chains[0][0:100].T, v11,v12, 10,10, border_size = 1,normalized = False), 'data') 
VIS.imshow_matrix(VIS.tile_matrix_rows(layers[0].activation(None,pos_chains[1])[0][0:100].T, v11,v12, 10,10, border_size = 1,normalized = False), 'sample 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(pos_chains[1][0:100].T, v21,v22, 10,10, border_size = 1,normalized = False), 'sample 2')
VIS.imshow_matrix(VIS.tile_matrix_rows(neg_chains[0][0:100].T, v11,v12, 10,10, border_size = 1,normalized = False), 'datan') 
VIS.imshow_matrix(VIS.tile_matrix_rows(layers[0].activation(None,neg_chains[1])[0][0:100].T, v11,v12, 10,10, border_size = 1,normalized = False), 'sample 1n')
VIS.imshow_matrix(VIS.tile_matrix_rows(neg_chains[1][0:100].T, v21,v22, 10,10, border_size = 1,normalized = False), 'sample 2n')
#VIS.imshow_matrix(VIS.tile_matrix_rows(z_e[0:100].T, v31,v32, 10,10, border_size = 1,normalized = False), 'sample 3')
#VIS.imshow_matrix(VIS.tile_matrix_rows(a_e[0:100].T, v41,v42, 10,10, border_size = 1,normalized = False), 'sample 4')
# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(layers[0].output_weight_layer.weights, v11,v12, v21,v22, border_size = 1,normalized = False), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(layers[0].output_weight_layer.weights,layers[1].output_weight_layer.weights), v11,v12, v31,v32, border_size = 1,normalized = False), 'Weights 2')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(layers[0].output_weight_layer.weights,numx.dot(layers[1].output_weight_layer.weights,layers[2].output_weight_layer.weights)), v11,v12, v41,v42, border_size = 1,normalized = False), 'Weights 3')
#VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(layers[0].output_weight_layer.weights,numx.dot(layers[1].output_weight_layer.weights,numx.dot(layers[2].output_weight_layer.weights,layers[3].output_weight_layer.weights))), v11,v12, v41,v42, border_size = 1,normalized = False), 'Weights 4')
#VIS.imshow_matrix(VIS.tile_matrix_rows(layers[2].output_weight_layer.weights, v31,v32, v41,v42, border_size = 1,normalized = False), 'Weights 4')



    
VIS.show()
"""