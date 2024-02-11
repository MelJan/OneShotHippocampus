import numpy as numx
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.misc.measuring as MEASURE
import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO
from scipy.signal import convolve2d
from sklearn.linear_model import LogisticRegression

# Set the same seed value for all algorithms
numx.random.seed(42)

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
        # Update weights
        self.weights += weight_updates
        
        # Restricts the gradient
        if numx.isscalar(restriction):
            if restriction > 0:
                if restriction_typ is 'Cols':
                    self.weights = numxExt.restrict_norms(
                                                    self.weights,
                                                    restriction, 0 )
                if restriction_typ is 'Rows':
                    self.weights = numxExt.restrict_norms(
                                                    self.weights,
                                                    restriction, 1 )
                if restriction_typ is 'Mat':
                    self.weights = numxExt.restrict_matrix_norm(
                                                    self.weights,
                                                    restriction )
                    
                if restriction_typ is 'Val':
                    self.weights = numx.clip(self.weights, -restriction, restriction,self.weights)
        
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

        if data is not None:
            if isinstance(data,list):
                data = numx.concatenate(data)
            if self.input_dim != data.shape[1]:
                raise ValueError("Data dimension and model input \
                                     dimension have to be equal!")  
            data_mean = numx.mean(data,axis=0).reshape(1,data.shape[1])
            
        # AUTO   -> data != None -> Initialized to the data mean
        #           data == None -> Initialized to Visible range mean
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
        
        # AUTO   -> data != None -> Initialized to the inverse sigmoid of 
        #           data mean
        #           data == Initialized to randn()*0.01
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

    def activation(self, bottom_up_states, top_down_states):
        ''' Calculates the pre and post synaptic activation.
            
        :Parameters:
            bottom_up_states: activation comming from previous layer.
                             -type: numpy array [batch_size, input dim]
                     
            top_down_states:  activation comming from next layer.
                             -type: numpy array [batch_size, input dim]

        '''
        pre_act = 0.0
        if self.input_weight_layer != None:
            pre_act += self.input_weight_layer.propagate_up(bottom_up_states) 
        if self.output_weight_layer != None:
            pre_act += self.output_weight_layer.propagate_down(top_down_states)
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
    
    def update_biases(self, bias_updates):
        ''' This function updates the weight parameter.
            
        :Parameters:
            b_updates: Update for the bias parameter.
                      -type: numpy array [1, input dim]

        '''
        self.bias += bias_updates
        
    def update_offsets(self, offset_updates, shifting_factor):
        ''' This function updates the weight parameter.
            
        :Parameters:
            b_updates: Update for the bias parameter.
                      -type: numpy array [1, input dim]

        '''
        self.offset = (1.0-shifting_factor)*self.offset + shifting_factor*offset_updates



# Set the same seed value for all algorithms
numx.random.seed(42)

# Load Data
data = IO.load_mnist("../../../../../data/mnist.pkl.gz",False)[0]

# Set dimensions
v11 = v12 = 28
v21 = v22 = 10
v31 = v32 = 10

N = v11 * v12
M = v21 * v22
O = v31 * v32

wl1 = Weight_layer(input_dim = N, 
                   output_dim = M, 
                   initial_weights = 0.01,
                   dtype = numx.float64)
wl2 = Weight_layer(input_dim = M, 
                   output_dim = O, 
                   initial_weights = 0.01,
                   dtype = numx.float64)

l1 = Binary_layer(None, 
                  wl1, 
                  data = data, 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

l2 = Binary_layer(wl1, 
                  wl2, 
                  data = None, 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

l3 = Binary_layer(wl2, 
                  None, 
                  data = None, 
                  initial_bias = 'AUTO',
                  initial_offsets = 'AUTO',
                  dtype = numx.float64)

# Initialize parameters
max_epochs = 1
batch_size = 100
k_pos = 3
k_neg = 1

lr_W1 = 0.1
lr_W2 = 0.1

lr_b1 = 0.1
lr_b2 = 0.1
lr_b3 = 0.1

lr_o1 = 0.1
lr_o2 = 0.1
lr_o3 = 0.1

# Initialize negative Markov chain
x_m = numx.zeros((batch_size,v11*v12))+l1.offset
y_m = numx.zeros((batch_size,v21*v22))+l2.offset
z_m = numx.zeros((batch_size,v31*v32))+l3.offset

# Reparameterize RBM such that the inital setting is the same for centereing and centered training
l1.bias += numx.dot(0.0-l2.offset,wl1.weights.T)
l2.bias += numx.dot(0.0-l1.offset,wl1.weights) + numx.dot(0.0-l3.offset,wl2.weights.T)
l3.bias += numx.dot(0.0-l2.offset,wl2.weights)

# Start time measure and training
#measurer = MEASURE.Stopwatch()

for epoch in range(0,max_epochs) :
    for b in range(0,data.shape[0],batch_size):
        
        #print b/numx.float64(data.shape[0])
   
        #positive phase
        x_d = data[b:b+batch_size,:]
        y_d = numx.zeros((batch_size,M))+l2.offset
        z_d = numx.zeros((batch_size,O))+l3.offset
        for _ in range(k_pos):
            y_d = l2.activation(x_d, z_d)[0]
            #y_d = l2.sample(y_d)
            z_d = l3.activation(y_d, None)[0]
            #z_d = l3.sample(z_d) 

        #negative phase
        for _ in range(k_neg):
            y_m = l2.activation(x_m, z_m)
            y_m = l2.sample(y_m)
            x_m = l1.activation(None,y_m)
            x_m = l1.sample(x_m)  
            z_m = l3.activation(y_m, None)
            z_m = l3.sample(z_m) 
        
        # Estimate new means
        new_o1 = numx.mean(x_d,axis=0)
        new_o2 = numx.mean(y_d,axis=0)
        new_o3 = numx.mean(z_d,axis=0)

        l1.update_offsets(numx.mean(x_d,axis = 0).reshape(1,x_d.shape[1]), lr_o1)
        l2.update_offsets(numx.mean(y_d,axis = 0).reshape(1,y_d.shape[1]), lr_o2)
        l3.update_offsets(numx.mean(z_d,axis = 0).reshape(1,z_d.shape[1]), lr_o3)

        # Calculate normal gradients
        wl1_grad = wl1.calculate_weight_gradients(x_d, y_d, x_m, y_m, l1.offset, l2.offset)
        wl2_grad = wl2.calculate_weight_gradients(y_d, z_d, y_m, z_m, l2.offset, l3.offset)
        
        # Calsulate centered gradients for biases and update
        grad_b1 =l1.calculate_gradient_b(x_d, x_m, None, l2.offset, None, wl1_grad)
        grad_b2 =l2.calculate_gradient_b(y_d, y_m, l1.offset, l3.offset, wl1_grad, wl2_grad)
        grad_b3 =l3.calculate_gradient_b(z_d, z_m, l2.offset, None, wl2_grad, None)

        wl1.update_weights(lr_W1*wl1_grad, None, None)
        wl2.update_weights(lr_W2*wl2_grad, None, None)

        #Einbauen letzten zustande saven in Unit layer
        l1.update_biases(lr_b1*grad_b1)
        l2.update_biases(lr_b2*grad_b2)
        l3.update_biases(lr_b3*grad_b3)
        
        print numx.mean(numxExt.get_norms(wl1.weights)),'\t',numx.mean(numxExt.get_norms(wl2.weights)),'\t',
        print numx.mean(numxExt.get_norms(l1.bias)),'\t',numx.mean(numxExt.get_norms(l2.bias)),'\t',
        print numx.mean(numxExt.get_norms(l3.bias)),'\t',numx.mean(l1.offset),'\t',numx.mean(l2.offset),'\t',numx.mean(l3.offset)
        

# Show weights
VIS.imshow_matrix(VIS.tile_matrix_rows(wl1.weights, v11,v12, v21,v22, border_size = 1,normalized = False), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(wl1.weights,wl2.weights), v11,v12, v31,v32, border_size = 1,normalized = False), 'Weights 2')

# Show some sampling
x_m = x_d
z_m = l3.offset
VIS.imshow_matrix(VIS.tile_matrix_columns(x_d, v11, v12, 1, batch_size, 1, False),'d')
for i in range(4):
    y_m = l2.activation(x_m, z_m)
    y_m = l2.sample(y_m)
    x_m = l1.activation(None,y_m)
    VIS.imshow_matrix(VIS.tile_matrix_columns(x_m[0], v11, v12, 1, batch_size, 1, False),'m '+str(i))
    x_m = l1.sample(x_m)  
    z_m = l3.activation(y_m, None)
    z_m = l3.sample(z_m) 
    
            
VIS.show()