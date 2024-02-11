__docformat__ = "restructuredtext en"

import mdp
from mdp import numx

import pydeep.rbm.model as MODEL
import pydeep.rbm.trainer as TRAINER
import pydeep.rbm.estimator as ESTIMATOR


class BinaryBinaryRBMNode(mdp.Node):
    ''' Implementation of a restricted Boltzmann machine with binary visible
        and binary hidden units.     

    '''
  
    def __init__(self, 
                  number_visibles, 
                  number_hiddens,  
                  data = None, 
                  initial_weights = 'AUTO', 
                  initial_visible_bias = 'AUTO', 
                  initial_hidden_bias = 'AUTO', 
                  initial_visible_mean = 'AUTO', 
                  initial_hidden_mean = 'AUTO', 
                  rbm_copy = None, 
                  dtype = numx.float64):
        ''' This function initializes all necessary parameters and data 
            structures. It is recommended to pass the training data to 
            initialize the network automatically.
            
        :Parameters:
            number_visibles:      Number of the visible variables.
                                 -type: int
                                  
            number_hiddens        Number of hidden variables.
                                 -type: int
                                  
            data:                 The training data for parameter 
                                  initialization if 'AUTO' is chosen.
                                 -type: None or 
                                        numpy array [num samples, input dim]
                                        
            initial_weights:      Initial weights.
                                  'AUTO' is random
                                 -type:  'AUTO', scalar or 
                                         numpy array [input dim, output_dim]
                                  
            initial_visible_bias: Initial visible bias.
                                  'AUTO' is random
                                  'INVERSE_SIGMOID' is the inverse Sigmoid of 
                                   the visilbe mean
                                 -type:  'AUTO','INVERSE_SIGMOID', scalar or 
                                         numpy array [1, input dim]
                                  
            initial_hidden_bias:  Initial hidden bias.
                                  'AUTO' is random
                                  'INVERSE_SIGMOID' is the inverse Sigmoid of 
                                   the hidden mean
                                 -type:  'AUTO','INVERSE_SIGMOID', scalar or 
                                         numpy array [1, output_dim]
                                  
            initial_visible_mean: Initial visible mean values.
                                  AUTO = data mean or 0.5 if not data is given.
                                 -type:  'AUTO', scalar or 
                                         numpy array [1, input dim]
                                  
            initial_hidden_mean:  Initial hidden mean values.
                                  AUTO = 0.5
                                 -type: 'AUTO', scalar or 
                                         numpy array [1, output_dim]
                                 
            rbm_copy:             If not None the rbm is initialized to the 
                                  values of the given RBM, then all other 
                                  given values will be overwritten.
                                 -type: None or BinaryBinaryRBM object
                        
            dtype:                Used data type i.e. numpy.float64
                                 -type: numpy.float32 or numpy.float64 or 
                                        numpy.float128  
            
        '''
        # If SKIP is set the model will not be initialized
        if rbm_copy != 'SKIP':
            self.model = MODEL.BinaryBinaryRBM(number_visibles, 
                                               number_hiddens,  
                                               data, 
                                               initial_weights, 
                                               initial_visible_bias, 
                                               initial_hidden_bias, 
                                               initial_visible_mean, 
                                               initial_hidden_mean, 
                                               dtype)
            super(BinaryBinaryRBMNode, self).__init__(self.model.input_dim
                                                      ,self.model.output_dim
                                                      ,self.model.dtype) 
        else:
            super(BinaryBinaryRBMNode, self).__init__(number_visibles
                                                      ,number_hiddens
                                                      ,dtype) 
        self.V_BINARY = True
        self.H_BINARY = True
        self._method = None
        self._trainer = None
        
    def get_parameters(self):
        ''' This function returns all model parameters as a list.
        
        :Returns: 
            The parameter references in a list.
           -type: list 

        ''' 
        return self.model.get_parameters()

    def sample_v(self, 
                  prob_v_given_h):
        ''' Samples the visible variables from the 
            conditional probabilities v given h.  
        
        :Parameters:
            prob_v_given_h: Conditional probabilities of v given h.
                           -type: numpy array [batch size, input dim]
                            
        :Returns: 
            States for prob_v_given_h.
           -type: numpy array [batch size, input dim]
        
        '''
        self._check_input(prob_v_given_h)
        return self.model.sample_v(prob_v_given_h)

    def probability_v_given_h(self, 
                                 h):
        ''' Calculates the conditional probabilities of v given h.      
        
        :Parameters:
            h:     Hidden states.
                  -type: numpy array [batch size, output dim]
                   
        :Returns: 
            Conditional probabilities v given h.
           -type: numpy array [batch size, input dim]
        
        '''
        self._check_output(h)
        return self.model.probability_v_given_h(h, 1.0)
    
    def log_probability_v(self, 
                            logZ, 
                            v):
        ''' Computes the log-probability / LogLikelihood(LL) for the given 
            visible units for this model. 
            To estimate the LL we need to know the logarithm of the partition 
            function Z. For small models it is possible to calculate Z, however
            since this involves calculating all possible hidden states, it is 
            intractable for bigger models. As an estimation method annealed 
            importance sampling (AIS) can be used instead.
        
        :Parameters:
            logZ: The logarithm of the partition function.
                 -type: float
            
            v:    Visible states.
                 -type: numpy array [batch size, input dim]
            
        :Returns:
            Log probability for v.
           -type: numpy array [batch size, 1]
            
        '''
        self._check_input(v)
        return self.model.log_probability_v(logZ, v, 1.0)
                                                  
    def sample_h(self, 
                  prob_h_given_v):
        ''' Samples the hidden variables from the 
            conditional probabilities h given v.
        
        :Parameters:
            prob_h_given_v: Conditional probabilities of h given v.
                           -type: numpy array [batch size, output dim]
        :Returns: 
            States for prob_h_given_v.
           -type: numpy array [batch size, output dim]
            
        '''
        self._check_output(prob_h_given_v)
        return self.model.sample_h(prob_h_given_v)
     
    def probability_h_given_v(self, 
                                 v):
        ''' Calculates the conditional probabilities of h given v.      
        
        :Parameters:
            v:     Visible states / data.
                  -type: numpy array [batch size, input dim]
        
        :Returns: 
            Conditional probabilities h given v.
           -type: numpy array [batch size, output dim]
        
        '''
        self._check_input(v)
        return self.model.probability_h_given_v(v, 1.0)

    def log_probability_h(self, 
                            logZ, 
                            h):
        ''' Computes the log-probability / LogLikelihood(LL) for the given 
            hidden units for this model. 
            To estimate the LL we need to know the logarithm of the partition 
            function Z. For small models it is possible to calculate Z, however
            since this involves calculating all possible hidden states, it is 
            intractable for bigger models. As an estimation method annealed 
            importance sampling (AIS) can be used instead.
        
        :Parameters:
            logZ: The logarithm of the partition function.
                 -type: float
            
            h:    Hidden states.
                 -type: numpy array [batch size, output dim]
            
        :Returns:
            Log probability for h.
           -type: numpy array [batch size, 1]
            
        '''
        self._check_output(h)
        return self.model.log_probability_h(logZ, h, 1.0)

    def energy(self, 
                v, 
                h):
        ''' Compute the energy of the RBM given observed variable states v
            and hidden variables state h.
        
        :Parameters:
            v:     The visible units states.
                  -type: numpy array [batch size, input dim]
            
            h:     The hidden units states.
                  -type: numpy array [batch size, output dim]
                  
        :Returns:
            Energy of v and h.
           -type: numpy array [batch size,1]
        
        '''
        self._check_input(v)
        self._check_output(h)
        return self.model.energy(v, h, 1.0)

    def log_probability_v_h(self, 
                               logZ, 
                               v, 
                               h):
        ''' Computes the joint log-probability / LogLikelihood(LL) for the 
            given visible and hidden units for this model. 
            To estimate the LL we need to know the logarithm of the partition 
            function Z. For small models it is possible to calculate Z, however
            since this involves calculating all possible hidden states, it is 
            intractable for bigger models. As an estimation method annealed 
            importance sampling (AIS) can be used instead.
         
        :Parameters:
            logZ: The logarithm of the partition function.
                 -type: float
            
            v:    Visible states.
                 -type: numpy array [batch size, input dim]
                 
            h:    Hidden states.
                 -type: numpy array [batch size, output dim]
            
        :Returns:
            Joint log probability for v and h.
           -type: numpy array [batch size, 1]
            
        '''
        self._check_input(v)
        self._check_output(h)
        return self.model.log_probability_v_h(logZ, v, h, 1.0)

    def reconstruction_error(self,
                                v, 
                                k=1, 
                                use_states=False, 
                                absolut_error = False):
        ''' This function calculates the reconstruction errors for a given 
            model and data.         
            
        :Parameters:
            model          The model
                          -type: Valid RBM model
                         
            v:             The data as 2D array.
                          -type: numpy array [num samples, num dimensions]
                         
            k:             Number of Gibbs sampling steps.
                          -type: int
                         
            use_states:    If false (default) the probabilities are used as 
                           reconstruction, if true states are sampled.
                          -type: bool
    
            absolut_error: If false (default) the squared error is used, the 
                           absolute error otherwise 
                          -type: bool
            
        :Returns:
            Reconstruction errors of the data.
           -type: nump array [num samples]
                 
        ''' 
        self._check_input(v)
        return ESTIMATOR.reconstruction_error(self.model, 
                                    v, 
                                    k, 
                                    1.0, 
                                    use_states, 
                                    absolut_error)

    def calculate_partition_function(self, 
                                         verbose=False):
        ''' Computes the true partition function for the given model.
            This is only possible if at least one layer is binary and
            small enought i.e. < 25 units.
           
        :Info:
            Exponential increase of computations by the number of visible 
            units. (16 usually ~ 20 seconds)
            
        :Parameters:
            verbose   If true prints the progress to the console.
                      -type: bool
        
        :Returns:
            Log Partition function for the model.
           -type: float
            
        ''' 
        layer = None
        # Binary RBM  
        if self.V_BINARY and self.H_BINARY:
            # Chose the smaller layer
            if self.model.input_dim < self.model.output_dim:
                layer = 'V'
            else:
                layer = 'H'
        else:
            if self.V_BINARY:
                layer = 'V'
            else:
                if self.H_BINARY:
                    layer = 'H'
                    
        lnZ = None          
        if layer is 'H':
            if self.model.output_dim >= 25:
                raise mdp.MDPWarning('For calculating the partition function '
                                    +'at least one layer needs to have just a '
                                    +'few binary units (i.e. < 25). Use the '
                                    +'approximation method instead!')
            lnZ = ESTIMATOR.partition_function_factorize_h(self.model, 
                                                           1.0, 
                                                           'AUTO', 
                                                           verbose)
        else:
            if layer is 'V':
                if self.model.output_dim >= 25:
                    raise mdp.MDPWarning('For calculating the partition' 
                                        +'function at least one layer needs '
                                        +'to have just a few binary units '
                                        +'(i.e. < 25). Use the approximation '
                                        +' method instead!')
                lnZ = ESTIMATOR.partition_function_factorize_v(self.model, 
                                                               1.0, 
                                                               'AUTO', 
                                                               verbose)
            else:
                raise mdp.MDPWarning('For calculating the partition function '
                                    +'at least one layer needs to have just a '
                                    +'few binary units (i.e. < 25). Use the '
                                    +'approximation method instead!')
        return lnZ
                
    def approximate_partition_function(self, 
                                           num_chains = 100, 
                                           betas = 10000, 
                                           verbose=False):
        ''' Approximates the partition function for the given model using 
            annealed importance sampling.
        
        :Parameters:
            num_chains: Number of AIS runs.
                       -type: int
            
            betas:      Number or a list of inverse temperatures to sample 
                        from.
                       -type: int, numpy array [num_betas]
            
            verbose:    If true prints the progress on console.
                       -type: bool
            
        
        :Returns:
            Mean estimated log partition function.
           -type: float
            Minimal estimated log partition function.
           -type: float
            Maximal estimated log partition function.
           -type: float
            Standard deviation of the estimation.
           -type: float
        
        '''   
        return ESTIMATOR.annealed_importance_sampling(self.model, 
                                                      num_chains, 
                                                      1, 
                                                      betas, 
                                                      verbose)
              
    def _execute(self, v, return_probs=True):
        ''' Samples the hidden variables from the 
            conditional probabilities h given v.
        
        :Parameters:
            v:     Visible states.
                  -type: numpy array [batch size, input dim]
        :Returns: 
            Probs or states for the hiddens.
           -type: numpy array [batch size, output dim]
            
        '''
        self._pre_execution_checks(v)
        prob_h_given_v = self.model.probability_h_given_v(v, 1.0)
        if not return_probs:
            prob_h_given_v = self.model.sample_h(prob_h_given_v)
        return prob_h_given_v
        
    def _inverse(self, h, return_probs=True):
        ''' Samples the visible variables from the 
            conditional probabilities v given h.
        
        :Parameters:
            h:     Hidden states.
                  -type: numpy array [batch size, output dim]
        :Returns: 
            Probs or states for visibles.
           -type: numpy array [batch size, input dim]
            
        '''
        self._pre_inversion_checks(h)
        prob_v_given_h = self.model.probability_v_given_h(h, 1.0)
        if not return_probs:
            prob_v_given_h = self.model.sample_v(prob_v_given_h)
        return prob_v_given_h

    def _set_training_method(self, method):
        ''' This function sets the training method. 
            The syntax is as follows:
            
            method:                Meaning:
            CD                    Contrastive Divergence
            PCD-int               Persitent Contrastive Divergence
                                  the integer value sets the number 
                                  of chains i.e. the batch_size.
            PT-int                Parallel Tempering Contrastive 
                                  Divergence. The integer value 
                                  sets the number of temperatures 
                                  to use.
            IPT-int-int           Independent Parallel Tempering 
                                  Contrastive Divergence. The first 
                                  integer value sets the number of 
                                  chains to use i.e. the batch_size
                                  and the second integer sets the 
                                  number of temperatures to use.
            GD                    Gradient decent, only possible 
                                  for small BinaryBinary RBMs.
            
        :Parameters:
            method  The description of the method.
                   -type: string
                 
        ''' 
        values = method.split('-')
        try:
            if values[0] is 'CD':
                self.trainer = TRAINER.CD(self.model)
            elif values[0] is 'PCD':
                self.trainer = TRAINER.PCD(self.model, int(values[1]))
            elif values[0] is 'PT':
                self.trainer = TRAINER.PT(self.model, int(values[1]))
            elif values[0] is 'IPT':
                self.trainer = TRAINER.IPT(self.model, int(values[1]), int(values[2]))
            elif values[0] is 'GD':
                if not self.V_BINARY or not self.H_BINARY:
                    raise mdp.MDPException('The True Gradient is only possible'
                                           +' for BinaryBinary RBMs')
                if self.input_dim > 20 or self.output_dim> 20:
                    raise mdp.MDPWarning('Training will be extremly slow, the '
                                         +'number of visible and hidden units '
                                         +'need to be small.')
                self.trainer = TRAINER.GD(self.model)
            else:
                raise mdp.MDPException('Invalid method description. Syntax CD'
                                       +', PCD-int, PT-int, IPT-int-int, GD]')
        except:
            raise mdp.MDPException('Invalid method description. Syntax CD'
                                    +', PCD-int, PT-int, IPT-int-int, GD]')
        self._method = method
            
    def _train(self, 
                v,
                epsilon=0.1,  
                momentum=0.5, 
                weight_decay=0.0, 
                k = 1,
                method = 'PT-5', 
                update_visible_mean = 0.0,
                update_hidden_mean = 0.0, 
                desired_hidden_activity = None, 
                restrict_gradient = False, 
                use_hidden_states = False):
        ''' Train the model, to achieve optimal performance 
            the number of samples (v.shape[0]) should not 
            change.
        
        :Parameters:
            v:                       The data used for training.
                                    -type: list of numpy arrays 
                                           [num samples, input dimension]
            
            epsilon:                 The learning rate. A scalar sets the 
                                     learning for all parameters to the same 
                                     value. Good value is often 0.01
                                    -type: float, numpy array [num parameters] 
            
            k:                       The number of Gibbs sampling steps.
                                     Good value if functionality is used: 
                                     The bigger the better but also 
                                     computationally more expensive.
                                    -type: int

            method:                  String that describes the training method.
                                     .. seealso:: _set_training_method()
                                    -type: string
            
            momentum:                The momentum term. A scalar sets the 
                                     momentum for all parameters to the same 
                                     value. 
                                     Good value if functionality is used: 0.9.
                                    -type: float , numpy array [num parameters]
            
            weight_decay:            The decay term. A scalar sets the decay 
                                     value for all 
                                     parameters to the same value.
                                     Good value if functionality is used:0.001
                                    -type: float , numpy array [num parameters]

            update_visible_mean:     The update step size for the models
                                     visible mean.
                                     Good value if functionality is used: 0.001
                                    -type: float
                                     
            update_hidden_mean:      The update step size for the models hidden
                                     mean.
                                     Good value if functionality is used: 0.001
                                    -type: float

            desired_hidden_activity: Desired average hidden activation or None
                                     for no regularization. Good value if
                                     functionality is used: 0.05.
                                    -type: float or None
                                   
            restrict_gradient:       If a scalar is given the norm of the 
                                     weight gradient is restricted to stay 
                                     below this value.   
                                    -type: None, float
                                        
            use_hidden_states:       If True, the hidden states are used for 
                                     the gradient calculations, the hiddens 
                                     probabilities otherwise. Adds noise on the
                                     gradient, helps for training GB-RBMs.
                                    -type: bool

        ''' 

        self._check_input(v)
        # Trainer is switched?
        if method != self._method:
            self._set_training_method(method)

        self.trainer.train(v,
                           1,
                           epsilon, 
                           k,  
                           momentum, 
                           weight_decay, 
                           update_visible_mean,
                           update_hidden_mean, 
                           desired_hidden_activity, 
                           restrict_gradient, 
                           use_hidden_states,
                           False)
        
class GaussianBinaryRBMNode(BinaryBinaryRBMNode):
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
                  initial_visible_mean=0.0, 
                  initial_hidden_mean=0.0,
                  rbm_copy=None, 
                  dtype=numx.float64):
        ''' This function initializes all necessary parameters and data 
            structures. See comments for automatically chosen values.
            
        :Parameters:
            number_visibles:      Number of the visible variables.
                                 -type: int
                                  
            number_hiddens        Number of hidden variables.
                                 -type: int
                                  
            data:                 The training data for initializing the 
                                  visible bias.
                                 -type: None or 
                                        numpy array [num samples, input dim]
                                        or List of numpy arrays
                                        [num samples, input dim]
            
            initial_weights:      Initial weights.
                                 -type: 'AUTO', scalar or 
                                        numpy array [input dim, output_dim]
                                  
            initial_visible_bias: Initial visible bias.
                                 -type: 'AUTO', scalar or 
                                        numpy array [1,input dim]
                                  
            initial_hidden_bias:  Initial hidden bias.
                                 -type: 'AUTO', scalar or 
                                        numpy array [1, output_dim]
                                  
            initial_sigma:        Initial standard deviation for the model.
                                 -type: 'AUTO', scalar or 
                                        numpy array [1, input_dim]

            initial_visible_mean: Initial visible mean values.
                                 -type: 'AUTO', scalar or 
                                        numpy array [1, input dim]
                                  
            initial_hidden_mean:  Initial hidden mean values.
                                 -type: 'AUTO', scalar or 
                                        numpy array [1, output_dim]

            rbm_copy:             If not None, the rbm is initialized to the 
                                  values of the given RBM.
                                 -type: None or BinaryBinaryRBM object
                                                                    
            dtype:                Used data type.
                                 -type: numpy.float32, numpy.float64 and, 
                                        numpy.float128 
        '''
        super(GaussianBinaryRBMNode, self).__init__(number_visibles, 
                                                number_hiddens,
                                                rbm_copy = 'SKIP')
        # If SKIP is set the model will not be initialized
        if rbm_copy != 'SKIP':
            # override BinaryBinary model
            self.model = MODEL.GaussianBinaryRBM(number_visibles, 
                                             number_hiddens, 
                                             data,
                                             initial_weights,
                                             initial_visible_bias,
                                             initial_hidden_bias,
                                             initial_sigma,
                                             initial_visible_mean,
                                             initial_hidden_mean, 
                                             rbm_copy,
                                             dtype)
        self.V_BINARY = False
        self.H_BINARY = True
        
    def _train(self, 
                v,
                epsilon=0.1,  
                momentum=0.0, 
                weight_decay=0.0, 
                k = 1,
                method = 'CD-1', 
                update_visible_mean = 0.0,
                update_hidden_mean = 0.0, 
                desired_hidden_activity = 0.01, 
                restrict_gradient = False, 
                use_hidden_states = True):
        
        self._check_input(v)
        # Trainer is switched?
        if method != self._method:
            self._set_training_method(method)

        self.trainer.train(v,
                           1,
                           epsilon, 
                           k,  
                           momentum, 
                           weight_decay, 
                           update_visible_mean,
                           update_hidden_mean, 
                           desired_hidden_activity, 
                           restrict_gradient, 
                           use_hidden_states,
                           False)

class GaussianBinaryVarianceRBMNode(BinaryBinaryRBMNode):
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
                  initial_visible_mean=0.0, 
                  initial_hidden_mean=0.0,
                  rbm_copy=None, 
                  dtype=numx.float64):
        ''' This function initializes all necessary parameters and data 
            structures. See comments for automatically chosen values.
            
        :Parameters:
            number_visibles:      Number of the visible variables.
                                 -type: int
                                  
            number_hiddens        Number of hidden variables.
                                 -type: int
                                  
            data:                 The training data for initializing the 
                                  visible bias.
                                 -type: None or 
                                        numpy array [num samples, input dim]
                                        or List of numpy arrays
                                        [num samples, input dim]
            
            initial_weights:      Initial weights.
                                 -type: 'AUTO', scalar or 
                                        numpy array [input dim, output_dim]
                                  
            initial_visible_bias: Initial visible bias.
                                 -type: 'AUTO', scalar or 
                                        numpy array [1,input dim]
                                  
            initial_hidden_bias:  Initial hidden bias.
                                 -type: 'AUTO', scalar or 
                                        numpy array [1, output_dim]
                                  
            initial_sigma:        Initial standard deviation for the model.
                                 -type: 'AUTO', scalar or 
                                        numpy array [1, input_dim]

            initial_visible_mean: Initial visible mean values.
                                 -type: 'AUTO', scalar or 
                                        numpy array [1, input dim]
                                  
            initial_hidden_mean:  Initial hidden mean values.
                                 -type: 'AUTO', scalar or 
                                        numpy array [1, output_dim]

            rbm_copy:             If not None, the rbm is initialized to the 
                                  values of the given RBM.
                                 -type: None or BinaryBinaryRBM object
                                                                    
            dtype:                Used data type.
                                 -type: numpy.float32, numpy.float64 and, 
                                        numpy.float128 
        '''
        super(GaussianBinaryVarianceRBMNode, self).__init__(number_visibles, 
                                                        number_hiddens,
                                                        rbm_copy = 'SKIP')
        # If SKIP is set the model will not be initialized
        if rbm_copy != 'SKIP':
            # override BinaryBinary model
            self.model = MODEL.GaussianBinaryVarianceRBM(number_visibles, 
                                             number_hiddens, 
                                             data,
                                             initial_weights,
                                             initial_visible_bias,
                                             initial_hidden_bias,
                                             initial_sigma,
                                             initial_visible_mean,
                                             initial_hidden_mean, 
                                             rbm_copy,
                                             dtype)
