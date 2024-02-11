import numpy as numx
import pydeep.base.numpyextension as npExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.rbm.model as RBM_MODEL
import pydeep.rbm.sampler as RBM_SAMPLER
import pydeep.rbm.estimator as RBM_ESTIMATOR

class Trainer_PCD(object):
    
    def __init__(self, model, batch_size):
        
        # Set batch size
        self.batch_size = batch_size
        
        # Store model
        self.model = model
        
        rbm = RBM_MODEL.BinaryBinaryRBM(number_visibles = model.input_dim+model.hidden2_dim, 
                                             number_hiddens = model.hidden1_dim, 
                                             data=None, 
                                             initial_weights=numx.vstack((model.W1,model.W2.T)), 
                                             initial_visible_bias=numx.hstack((model.b1,model.b3)), 
                                             initial_hidden_bias=model.b2, 
                                             initial_visible_offsets=numx.hstack((model.o1,model.o3)), 
                                             initial_hidden_offsets=model.o2)
        
        # Initializee Markov chains
        self.m1 = model.o1+numx.zeros((batch_size,model.input_dim))
        self.m2 = model.o2+numx.zeros((batch_size,model.hidden1_dim))
        self.m3 = model.o3+numx.zeros((batch_size,model.hidden2_dim))

    def train(self, data, epsilon, k=[3,1], offset_typ = 'DDD',meanfield = False):
        
        #positive phase
        id1 = numx.dot(data-self.model.o1,self.model.W1)
        d3 = numx.copy(self.model.o3)
        d2 = 0.0
        #for _ in range(k[0]):  
        if meanfield == False:
            for _ in range(k[0]): 
                d3 = self.model.dtype(d3 > numx.random.random(d3.shape))
                d2 = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
                d2 = self.model.dtype(d2 > numx.random.random(d2.shape))
                d3 = Sigmoid.f(numx.dot(d2-self.model.o2,self.model.W2) + self.model.b3)
        else:
            if meanfield == True:
                for _ in range(k[0]): 
                    d2 = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
                    d3 = Sigmoid.f(numx.dot(d2-self.model.o2,self.model.W2) + self.model.b3)
            else:
                d2_new = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
                d3_new = Sigmoid.f(numx.dot(d2_new-self.model.o2,self.model.W2) + self.model.b3)
                while numx.max(numx.abs(d2_new-d2)) > meanfield or numx.max(numx.abs(d3_new-d3)) > meanfield: 
                    d2 = d2_new
                    d3 = d3_new
                    d2_new = Sigmoid.f( id1 + numx.dot(d3_new-self.model.o3,self.model.W2.T) + self.model.b2)
                    d3_new = Sigmoid.f(numx.dot(d2_new-self.model.o2,self.model.W2) + self.model.b3)
                d2 = d2_new
                d3 = d3_new
                
        #negative phase
        for _ in range(k[1]):  
            self.m2 = Sigmoid.f(numx.dot(self.m1-self.model.o1,self.model.W1) + numx.dot(self.m3-self.model.o3,self.model.W2.T) + self.model.b2)
            self.m2 = self.model.dtype(self.m2 > numx.random.random(self.m2.shape))
            self.m1 = Sigmoid.f(numx.dot(self.m2-self.model.o2,self.model.W1.T) + self.model.b1)
            self.m1 = self.model.dtype(self.m1 > numx.random.random(self.m1.shape))
            self.m3 = Sigmoid.f(numx.dot(self.m2-self.model.o2,self.model.W2) + self.model.b3)
            self.m3 = self.model.dtype(self.m3 > numx.random.random(self.m3.shape))
            
        # Estimate new means
        new_o1 = 0
        if offset_typ[0] is 'D':
            new_o1 = data.mean(axis=0)
        if offset_typ[0] is 'A':
            new_o1 = (self.m1.mean(axis=0)+data.mean(axis=0))/2.0
        if offset_typ[0] is 'M':
            new_o1 = self.m1.mean(axis=0)

        new_o2 = 0
        if offset_typ[1] is 'D':
            new_o2 = d2.mean(axis=0)
        if offset_typ[1] is 'A':
            new_o2 = (self.m2.mean(axis=0)+d2.mean(axis=0))/2.0
        if offset_typ[1] is 'M':
            new_o2 = self.m2.mean(axis=0)

        new_o3 = 0
        if offset_typ[2] is 'D':
            new_o3 = d3.mean(axis=0)
        if offset_typ[2] is 'A':
            new_o3 = (self.m3.mean(axis=0)+d3.mean(axis=0))/2.0
        if offset_typ[2] is 'M':
            new_o3 = self.m3.mean(axis=0)
             
        # Reparameterize
        self.model.b1 += epsilon[6]*numx.dot(new_o2-self.model.o2,self.model.W1.T)
        self.model.b2 += epsilon[5]*numx.dot(new_o1-self.model.o1,self.model.W1) + epsilon[7]*numx.dot(new_o3-self.model.o3,self.model.W2.T)
        self.model.b3 += epsilon[6]*numx.dot(new_o2-self.model.o2,self.model.W2)

        # Shift means
        self.model.o1 = (1.0-epsilon[5])*self.model.o1 + epsilon[5]*new_o1
        self.model.o2 = (1.0-epsilon[6])*self.model.o2 + epsilon[6]*new_o2
        self.model.o3 = (1.0-epsilon[7])*self.model.o3 + epsilon[7]*new_o3

        # Calculate gradients
        dW1 = (numx.dot((data-self.model.o1).T,d2-self.model.o2)-numx.dot((self.m1-self.model.o1).T,self.m2-self.model.o2))
        dW2 = (numx.dot((d2-self.model.o2).T,d3-self.model.o3)-numx.dot((self.m2-self.model.o2).T,self.m3-self.model.o3))
        
        db1 = (numx.sum(data-self.m1,axis = 0)).reshape(1,self.model.input_dim)
        db2 = (numx.sum(d2-self.m2,axis = 0)).reshape(1,self.model.hidden1_dim)
        db3 = (numx.sum(d3-self.m3,axis = 0)).reshape(1,self.model.hidden2_dim)

        # Update Model
        self.model.W1 += epsilon[0]/self.batch_size*dW1
        self.model.W2 += epsilon[1]/self.batch_size*dW2
        
        self.model.b1 += epsilon[2]/self.batch_size*db1
        self.model.b2 += epsilon[3]/self.batch_size*db2
        self.model.b3 += epsilon[4]/self.batch_size*db3

class TrainerCD(object):
    
    def __init__(self, model, batch_size):
        
        # Set batch size
        self.batch_size = batch_size
        
        # Store model
        self.model = model
        
        rbm = RBM_MODEL.BinaryBinaryRBM(number_visibles = model.input_dim+model.hidden2_dim, 
                                             number_hiddens = model.hidden1_dim, 
                                             data=None, 
                                             initial_weights=numx.vstack((model.W1,model.W2.T)), 
                                             initial_visible_bias=numx.hstack((model.b1,model.b3)), 
                                             initial_hidden_bias=model.b2, 
                                             initial_visible_offsets=numx.hstack((model.o1,model.o3)), 
                                             initial_hidden_offsets=model.o2)
        
        self.sampler = RBM_SAMPLER.Parallel_Tempering_sampler(rbm,20)

    def train(self, data, epsilon, k=[3,1], offset_typ = 'DDD',meanfield = False):
        
        #positive phase
        id1 = numx.dot(data-self.model.o1,self.model.W1)
        d3 = numx.copy(self.model.o3)
        d2 = 0.0
        #for _ in range(k[0]):  
        if meanfield == False:
            for _ in range(k[0]): 
                d3 = self.model.dtype(d3 > numx.random.random(d3.shape))
                d2 = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
                d2 = self.model.dtype(d2 > numx.random.random(d2.shape))
                d3 = Sigmoid.f(numx.dot(d2-self.model.o2,self.model.W2) + self.model.b3)
        else:
            if meanfield == True:
                for _ in range(k[0]): 
                    d2 = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
                    d3 = Sigmoid.f(numx.dot(d2-self.model.o2,self.model.W2) + self.model.b3)
            else:
                d2_new = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
                d3_new = Sigmoid.f(numx.dot(d2_new-self.model.o2,self.model.W2) + self.model.b3)
                while numx.max(numx.abs(d2_new-d2)) > meanfield or numx.max(numx.abs(d3_new-d3)) > meanfield: 
                    d2 = d2_new
                    d3 = d3_new
                    d2_new = Sigmoid.f( id1 + numx.dot(d3_new-self.model.o3,self.model.W2.T) + self.model.b2)
                    d3_new = Sigmoid.f(numx.dot(d2_new-self.model.o2,self.model.W2) + self.model.b3)
                d2 = d2_new
                d3 = d3_new
        
        self.sampler.model = RBM_MODEL.BinaryBinaryRBM(number_visibles = self.model.input_dim+self.model.hidden2_dim, 
                                             number_hiddens = self.model.hidden1_dim, 
                                             data=None, 
                                             initial_weights=numx.vstack((self.model.W1,self.model.W2.T)), 
                                             initial_visible_bias=numx.hstack((self.model.b1,self.model.b3)), 
                                             initial_hidden_bias=self.model.b2, 
                                             initial_visible_offsets=numx.hstack((self.model.o1,self.model.o3)), 
                                             initial_hidden_offsets=self.model.o2)
        sample = self.sampler.sample(self.batch_size, k[1])
        self.m2 = self.sampler.model.probability_h_given_v(sample)
        self.m1 = sample[:,0:self.model.input_dim]
        self.m3 = sample[:,self.model.input_dim:]
 
        # Estimate new means
        new_o1 = 0
        if offset_typ[0] is 'D':
            new_o1 = data.mean(axis=0)
        if offset_typ[0] is 'A':
            new_o1 = (self.m1.mean(axis=0)+data.mean(axis=0))/2.0
        if offset_typ[0] is 'M':
            new_o1 = self.m1.mean(axis=0)

        new_o2 = 0
        if offset_typ[1] is 'D':
            new_o2 = d2.mean(axis=0)
        if offset_typ[1] is 'A':
            new_o2 = (self.m2.mean(axis=0)+d2.mean(axis=0))/2.0
        if offset_typ[1] is 'M':
            new_o2 = self.m2.mean(axis=0)

        new_o3 = 0
        if offset_typ[2] is 'D':
            new_o3 = d3.mean(axis=0)
        if offset_typ[2] is 'A':
            new_o3 = (self.m3.mean(axis=0)+d3.mean(axis=0))/2.0
        if offset_typ[2] is 'M':
            new_o3 = self.m3.mean(axis=0)
             
        # Reparameterize
        self.model.b1 += epsilon[6]*numx.dot(new_o2-self.model.o2,self.model.W1.T)
        self.model.b2 += epsilon[5]*numx.dot(new_o1-self.model.o1,self.model.W1) + epsilon[7]*numx.dot(new_o3-self.model.o3,self.model.W2.T)
        self.model.b3 += epsilon[6]*numx.dot(new_o2-self.model.o2,self.model.W2)

        # Shift means
        self.model.o1 = (1.0-epsilon[5])*self.model.o1 + epsilon[5]*new_o1
        self.model.o2 = (1.0-epsilon[6])*self.model.o2 + epsilon[6]*new_o2
        self.model.o3 = (1.0-epsilon[7])*self.model.o3 + epsilon[7]*new_o3

        # Calculate gradients
        dW1 = (numx.dot((data-self.model.o1).T,d2-self.model.o2)-numx.dot((self.m1-self.model.o1).T,self.m2-self.model.o2))
        dW2 = (numx.dot((d2-self.model.o2).T,d3-self.model.o3)-numx.dot((self.m2-self.model.o2).T,self.m3-self.model.o3))
        
        db1 = (numx.sum(data-self.m1,axis = 0)).reshape(1,self.model.input_dim)
        db2 = (numx.sum(d2-self.m2,axis = 0)).reshape(1,self.model.hidden1_dim)
        db3 = (numx.sum(d3-self.m3,axis = 0)).reshape(1,self.model.hidden2_dim)

        # Update Model
        self.model.W1 += epsilon[0]/self.batch_size*dW1
        self.model.W2 += epsilon[1]/self.batch_size*dW2
        
        self.model.b1 += epsilon[2]/self.batch_size*db1
        self.model.b2 += epsilon[3]/self.batch_size*db2
        self.model.b3 += epsilon[4]/self.batch_size*db3

class Model(object):
    
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, offset_typ, data, dtype = numx.float64):
        ''' Initializes the network
        
        :Parameters:
            input_dim:    Number of input dimensions.
                         -type: int

            hidden1_dim:  Number of hidden dimensions for the first hidden layer.
                         -type: int
                 
            hidden2_dim:  Number of hidden dimensions for the first hidden layer.
                         -type: int
                         
            offset_typ:   Typs of offset values used for specific initialization
                          'DDD' -> Centering, 'AAA'-> Enhanced gradient,'MMM' -> Model mean centering
                         -type: string (3 chars)
                         
        ''' 
        # Set used data type
        self.dtype = dtype
        
        # Set dimensions
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        
        # Initialize weights
        self.W1 = numx.random.randn(input_dim, hidden1_dim) * 0.01
        self.W2 = numx.random.randn(hidden1_dim, hidden2_dim) * 0.01

        # Initialize offsets
        self.o1 = numx.zeros((1,input_dim)) 
        self.o2 = numx.zeros((1,hidden1_dim)) 
        self.o3 = numx.zeros((1,hidden2_dim))

        self.b1 = numx.zeros((1,input_dim)) 
        self.b2 = numx.zeros((1,hidden1_dim)) 
        self.b3 = numx.zeros((1,hidden2_dim))

        if data != None:
            datamean = numx.mean(data, axis = 0).reshape(1,input_dim)
            if offset_typ[0] is '0':
                self.b1 = Sigmoid.g(numx.clip(datamean,0.001,0.999))
            if offset_typ[0] is 'D':
                self.o1 = numx.copy(datamean)
                self.b1 = Sigmoid.g(numx.clip(self.o1,0.001,0.999))
            if offset_typ[0] is 'A':
                self.o1 = (datamean + 0.5)/2.0
                self.b1 = Sigmoid.g(numx.clip(self.o1,0.001,0.999))
            if offset_typ[0] is 'M':
                self.o1 += 0.5
        else:
            if offset_typ[0] != '0':
                self.o1 += 0.5

        if offset_typ[1] != '0':
            self.o2 += 0.5
            
        if offset_typ[2] != '0':
            self.o3 += 0.5

        
    def energy(self,x,h1,h2):
        ''' Computes the energy for x, h1 and h2.
        
        :Parameters:
            x:    Input layer states.
                 -type: numpy array [batch size, input dim]

            h1:   First layer states.
                 -type: numpy array [batch size, hidden1 dim]
                 
            h2:   Second layer states.
                 -type: numpy array [batch size, hidden2 dim]
                  
        :Returns:
            Energy for x, h1 and h2.
           -type: numpy array [batch size, 1]
            
        ''' 
        # centered variables
        xtemp = x-self.o1
        h1temp = h1-self.o2
        h2temp = h2-self.o3
        # Caluclate energy
        return - numx.dot(xtemp, self.b1.T)\
                - numx.dot(h1temp, self.b2.T) \
                - numx.dot(h2temp, self.b3.T) \
                - numx.sum(numx.dot(xtemp, self.W1) * h1temp,axis=1).reshape(h1temp.shape[0], 1)\
                - numx.sum(numx.dot(h1temp, self.W2) * h2temp,axis=1).reshape(h2temp.shape[0], 1)

    def unnormalized_log_probability_x(self,x):
        ''' Computes the unnormalized log probabilities of x.
        
        :Parameters:
            x:    Input layer states.
                 -type: numpy array [batch size, input dim]
                  
        :Returns:
            Unnormalized log probability of x.
           -type: numpy array [batch size, 1]
            
        '''  
        # Generate all possibel binary codes for h1 and h2
        all_h1 = npExt.generate_binary_code(self.W2.shape[0])
        all_h2 = npExt.generate_binary_code(self.W2.shape[1])
        # Center variables
        xtemp = x-self.o1
        h1temp = all_h1-self.o2
        h2temp = all_h2-self.o3
        # Bias term
        bias = numx.dot(xtemp, self.b1.T)
        # Both quadratic terms
        part1 = numx.exp(numx.dot(numx.dot(xtemp, self.W1)+self.b2, h1temp.T))
        part2 = numx.exp(numx.dot(numx.dot(h1temp, self.W2)+self.b3, h2temp.T))
        # Dot product of all combination of all quadratic terms + bias
        return bias+numx.log(numx.sum(numx.dot(part1,part2), axis = 1).reshape(x.shape[0],1))

    def unnormalized_log_probability_h1(self,h1):
        ''' Computes the unnormalized log probabilities of h1.
        
        :Parameters:
            h1:    First hidden layer states.
                  -type: numpy array [batch size, hidden1 dim]
                  
        :Returns:
            Unnormalized log probability of h1.
           -type: numpy array [batch size, 1]
            
        '''  
        # Centered
        temp = h1 - self.o2
        # Bias term
        bias = numx.dot(temp, self.b2.T).reshape(temp.shape[0], 1)
        # Value for h1 via factorization over x 
        activation = numx.dot(temp, self.W1.T) + self.b1
        factorx = numx.sum(
                           numx.log(
                                    numx.exp(activation*(1.0 - self.o1))
                                    + numx.exp(-activation*self.o1)
                                    ) 
                           , axis=1).reshape(temp.shape[0], 1)   
        # Value for h1 via factorization over h2
        activation = numx.dot(temp, self.W2) + self.b3  
        factorh2 = numx.sum(
                            numx.log(
                                     numx.exp(activation*(1.0 - self.o3))
                                     + numx.exp(-activation*self.o3)
                                     ) 
                            , axis=1).reshape(temp.shape[0], 1)  
        return bias + factorx + factorh2

    def unnormalized_log_probability_x_h2(self,x, h2):
        ''' Computes the unnormalized log probabilities of h1.
        
        :Parameters:
            x:     Input layer states.
                  -type: numpy array [batch size, input dim]        

            h2:    Second hidden layer states.
                  -type: numpy array [batch size, hidden2 dim]
                  
        :Returns:
            Unnormalized log probability of x, h2.
           -type: numpy array [batch size, 1]
            
        '''  
        # Centered
        tempx = x - self.o1
        temph2 = h2 - self.o3
        # Bias term
        bias = (numx.dot(tempx, self.b1.T)+numx.dot(temph2, self.b3.T)).reshape(tempx.shape[0], 1)
        # Value for h1 via factorization over x 
        activation = numx.dot(tempx, self.W1) + numx.dot(temph2, self.W2.T) + self.b2
        factorh1 = numx.sum(
                           numx.log(
                                    numx.exp(activation*(1.0 - self.o2))
                                    + numx.exp(-activation*self.o2)
                                    ) 
                           , axis=1).reshape(tempx.shape[0], 1)   
        # Value for h1 via factorization over h2
        return bias + factorx

class Estimator(object):

    @classmethod
    def _partition_function_exact_check(cls, model, batchsize_exponent='AUTO'):
        ''' Computes the true partition function for the given model by factoring 
            over the visible and hidden2 units.
            
            This is just proof of concept, use _partition_function_exact() instead, 
            it is heaps faster!
            
        :Parameters:
            model:              The model
                               -type: Valid DBM model
                               
            batchsize_exponent: 2^batchsize_exponent will be the batch size.
                               -type: int
        
        :Returns:
            Log Partition function for the model.
           -type: float
            
        '''  
        bit_length = model.W1.shape[1]
        if batchsize_exponent is 'AUTO' or batchsize_exponent > 20:
            batchsize_exponent = numx.min([model.W1.shape[1], 12])
        batchSize = numx.power(2, batchsize_exponent)
        num_combinations = numx.power(2, bit_length)
        num_batches = num_combinations / batchSize
        bitCombinations = numx.zeros((batchSize, model.W1.shape[1]))
        log_prob_vv_all = numx.zeros(num_combinations)

        for batch in range(1, num_batches + 1):
            # Generate current batch
            bitCombinations = npExt.generate_binary_code(bit_length, 
                                                         batchsize_exponent, 
                                                         batch - 1)
            # calculate LL
            log_prob_vv_all[(batch - 1) * batchSize:batch * batchSize] = model.unnormalized_log_probability_h1(bitCombinations).reshape(
                                                    bitCombinations.shape[0])
        # return the log_sum of values
        return npExt.log_sum_exp(log_prob_vv_all)

    @classmethod
    def partition_function_exact(cls, model, batchsize_exponent='AUTO'):
        ''' Computes the true partition function for the given model by factoring 
            over the visible and hidden2 units.
            
        :Parameters:
            model:              The model
                               -type: Valid DBM model
                   
            batchsize_exponent: 2^batchsize_exponent will be the batch size.
                               -type: int
        
        :Returns:
            Log Partition function for the model.
           -type: float
            
        '''    
        # We transform the DBM to an RBM with restricted connections.
        rbm = RBM_MODEL.BinaryBinaryRBM(number_visibles = model.input_dim+model.hidden2_dim, 
                                        number_hiddens = model.hidden1_dim, 
                                        data=None, 
                                        initial_weights=numx.vstack((model.W1,model.W2.T)), 
                                        initial_visible_bias=numx.hstack((model.b1,model.b3)), 
                                        initial_hidden_bias=model.b2, 
                                        initial_visible_offsets=numx.hstack((model.o1,model.o3)), 
                                        initial_hidden_offsets=model.o2)
        return RBM_ESTIMATOR.partition_function_factorize_h(rbm)

    @classmethod
    def partition_function_AIS(cls, model, num_chains = 100, k = 1, betas = 10000, status = False):
        ''' Approximates the partition function for the given model using annealed
            importance sampling.
        
            :Parameters:
                model:      The model.
                           -type: Valid RBM model.
                
                num_chains: Number of AIS runs.
                           -type: int
                
                k:          Number of Gibbs sampling steps.
                           -type: int
                
                beta:       Number or a list of inverse temperatures to sample from.
                           -type: int, numpy array [num_betas]
                
                status:     If true prints the progress on console.
                           -type: bool
                
            
            :Returns:
                Mean estimated log partition function.
               -type: float
                Mean +3std estimated log partition function.
               -type: float
                Mean -3std estimated log partition function.
               -type: float
        
        ''' 
        # We transform the DBM to an RBM with restricted connections.
        rbm = RBM_MODEL.BinaryBinaryRBM(number_visibles = model.input_dim+model.hidden2_dim, 
                                        number_hiddens = model.hidden1_dim, 
                                        data=None, 
                                        initial_weights=numx.vstack((model.W1,model.W2.T)), 
                                        initial_visible_bias=numx.hstack((model.b1,model.b3)), 
                                        initial_hidden_bias=model.b2, 
                                        initial_visible_offsets=numx.hstack((model.o1,model.o3)), 
                                        initial_hidden_offsets=model.o2)
        # Run AIS for the transformed DBM
        return RBM_ESTIMATOR.annealed_importance_sampling(model = rbm, 
                                                      num_chains =  num_chains, 
                                                      k = k, betas= betas, 
                                                      status = status)

    @classmethod
    def _LL_exact_check(cls, model, x, lnZ):
        ''' Computes the exact log likelihood for x by summing over all possible 
            states for h1, h2. Only possible for small hidden layers!
            
            This is just proof of concept, use LL_exact() instead, it is heaps faster!
            
        :Parameters:
            model:  The model
                   -type: Valid DBM model
                        
            x:      Input states.
                   -type: numpy array [batch size, input dim]
                 
            lnZ:    Logarithm of the patition function.
                   -type: float
                  
        :Returns:
            Exact log likelihood for x.
           -type: numpy array [batch size, 1]
            
        ''' 
        # Generate all binary codes
        all_h1 = npExt.generate_binary_code(model.W2.shape[0])
        all_h2 = npExt.generate_binary_code(model.W2.shape[1])
        result = numx.zeros(x.shape[0])
        for i in range(x.shape[0]):
            for j in range(all_h1.shape[0]):
                for k in range(all_h2.shape[0]):
                    result[i] += numx.exp(
                                          -model.energy(
                                                       x[i].reshape(1,x.shape[1]),
                                                       all_h1[j].reshape(1,all_h1.shape[1]),
                                                       all_h2[k].reshape(1,all_h2.shape[1]),
                                                       )
                                          )
        return numx.log(result) - lnZ

    @classmethod
    def LL_exact(cls, model, x, lnZ):
        ''' Computes the exact log likelihood for x by summing over all possible 
            states for h1, h2. Only possible for small hidden layers!
        
        :Parameters:
            model:  The model
                   -type: Valid DBM model
        
            x:     Input states.
                  -type: numpy array [batch size, input dim]
                 
            lnZ:   Logarithm of the patition function.
                  -type: float
                  
        :Returns:
            Exact log likelihood for x.
           -type: numpy array [batch size, 1]
            
        ''' 
        return model.unnormalized_log_probability_x(x)- lnZ

    @classmethod
    def _LL_lower_bound_check(cls, model, x, lnZ, conv_thres= 0.01, max_iter=100000):
        ''' Computes the log likelihood lower bound for x by approximating h1, h2 
            by Mean field estimates.
            .. seealso:: AISTATS 2009: Deep Bolzmann machines
                 http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS09_SalakhutdinovH.pdf
            
        :Parameters:
            model:       The model
                        -type: Valid DBM model
                        
            x:           Input states.
                        -type: numpy array [batch size, input dim]
                 
            lnZ:         Logarithm of the patition function.
                        -type: float

            conv_thres:  Convergence threshold for the mean field approximation
                        -type: float

            max_iter:    If convergence threshold not reached, maximal number of sampling steps
                        -type: int

        :Returns:
            Log likelihood lower bound for x.
           -type: numpy array [batch size, 1]
            
        '''
    
        
        # Pre calc activation from x since it is constant
        id1 = numx.dot(x-model.o1,model.W1)
        # Initialize mu3 with its mean
        d3 = 0.0
        d2 = 0.0
        # While convergence of max number of iterations not reached, 
        # run mean field estimation
        d2_new = Sigmoid.f( id1 + model.b2)
        d3_new = Sigmoid.f(numx.dot(d2_new-model.o2,model.W2) + model.b3)
        while numx.max(numx.abs(d2_new-d2)) > conv_thres or numx.max(numx.abs(d3_new-d3)) > conv_thres: 
            d2 = d2_new
            d3 = d3_new
            d2_new = Sigmoid.f( id1 + numx.dot(d3_new-model.o3,model.W2.T) + model.b2)
            d3_new = Sigmoid.f(numx.dot(d2_new-model.o2,model.W2) + model.b3)
        d2 = numx.clip(d2_new,0.0000000000000001,0.9999999999999999)
        d3 = numx.clip(d3_new,0.0000000000000001,0.9999999999999999)
        # Return ernegy of states + the entropy of h1.h2 due to the mean field approximation
        return -model.energy(x,d2,d3) -lnZ - numx.sum(d2*numx.log(d2)+(1.0-d2)*numx.log(1.0-d2),axis = 1).reshape(x.shape[0], 1) - numx.sum(d3*numx.log(d3)+(1.0-d3)*numx.log(1.0-d3),axis = 1).reshape(x.shape[0], 1)

    @classmethod
    def LL_lower_bound(cls, model, x, lnZ, conv_thres= 0.00000000001, max_iter=1000):
        ''' Computes the log likelihood lower bound for x by approximating h1 
            by Mean field estimates. The same as LL_lower_bound, but where h2 has been factorized.
            .. seealso:: AISTATS 2009: Deep Bolzmann machines
                 http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS09_SalakhutdinovH.pdf
            
        :Parameters:
            model:       The model
                        -type: Valid DBM model
        
            x:           Input states.
                        -type: numpy array [batch size, input dim]
                 
            lnZ:         Logarithm of the patition function.
                        -type: float

            conv_thres:  Convergence threshold for the mean field approximation
                        -type: float
                        
            max_iter:    If convergence threshold not reached, maximal number of sampling steps
                        -type: int
                  
        :Returns:
            Log likelihood lower bound for x.
           -type: numpy array [batch size, 1]
            
        '''
        # Pre calc activation from x since it is constant
        id1 = numx.dot(x-model.o1,model.W1)
        # Initialize mu3 with its mean
        d3 = 0.0
        d2 = 0.0
        # While convergence of max number of iterations not reached, 
        # run mean field estimation
        d2_new = Sigmoid.f( id1 + model.b2)
        d3_new = Sigmoid.f(numx.dot(d2_new-model.o2,model.W2) + model.b3)
        while numx.max(numx.abs(d2_new-d2)) > conv_thres: 
            d2 = d2_new
            d3 = d3_new
            d2_new = Sigmoid.f( id1 + numx.dot(d3_new-model.o3,model.W2.T) + model.b2)
            d3_new = Sigmoid.f(numx.dot(d2_new-model.o2,model.W2) + model.b3)
            
        d2 = numx.clip(d2_new,0.0000000000000001,0.9999999999999999)
            
        # Foactorize over h2
        xtemp = x-model.o1
        h1temp = d2-model.o2
        e2 = numx.prod(numx.exp(-(numx.dot(h1temp, model.W2)+model.b3)*(model.o3))+numx.exp((numx.dot(h1temp, model.W2)+model.b3)*(1.0-model.o3)), axis = 1).reshape(x.shape[0],1)
        e1 =  numx.dot(xtemp, model.b1.T)\
            + numx.dot(h1temp, model.b2.T) \
            + numx.sum(numx.dot(xtemp, model.W1) * h1temp ,axis=1).reshape(h1temp.shape[0], 1) + numx.log(e2)
        # Return energy of states + the entropy of h1 due to the mean field approximation
        return e1-lnZ - numx.sum(d2*numx.log(d2) + (1.0-d2)*numx.log(1.0-d2) ,axis = 1).reshape(x.shape[0], 1)
  