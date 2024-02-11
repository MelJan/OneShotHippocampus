import numpy as numx
import pydeep.base.numpyextension as numxExt

class DBM_model(object):
    
    def __init__(self, layers):
        ''' This function initializes the weight layer.
            
        :Parameters:
            model:      Model, basically a list of layers
                       -type: Weight_layer or None
                                  
            dtype:      Used data type i.e. numpy.float64
                       -type: numpy.float32 or numpy.float64 or 
                                           numpy.float128  
            
        '''  
        self.layers = layers
        self.num_layers = len(layers)
        if self.num_layers % 2 == 0:
            self.even_num_layers = True
        else:
            self.even_num_layers = False

    def meanfield_estimate(self, data, k, target = None):

        # Create empty list with pre caluclated values
        pre_top = []
        pre_bottom = []
        for i in range(self.num_layers):
            pre_top.append(None)
            pre_bottom.append(None)

        # Get all pre calculatable values
        for i in range(self.num_layers):
            if data[i] != None:
                if i > 0:
                    pre_top[i-1] = self.layers[i].input_weight_layer.propagate_down(data[i])
                if i < self.num_layers-1:
                    pre_bottom[i+1] = self.layers[i].output_weight_layer.propagate_up(data[i])
    
        # Set all determined values
        if target == None:
            target = []
            for i in range(0,self.num_layers):
                if data[i] != None:
                    target.append(data[i])
                else:
                    target.append(numx.zeros((data[0].shape[0],self.layers[i].input_dim))+self.layers[i].offset)
        else:
            for i in range(0,self.num_layers):
                target[i] = data[i]

        # performa mf updates only for not fixed values
        for i in range(k):
            for i in range(1,self.num_layers-1,2):
                if data[i] == None:
                    target[i] = self.layers[i].activation(target[i-1], target[i+1], pre_bottom[i], pre_top[i])[0]

            #Top layer if even
            if self.even_num_layers == True:
                if data[self.num_layers-1] == None:
                    target[self.num_layers-1] = self.layers[self.num_layers-1].activation(target[self.num_layers-2], None, pre_bottom[self.num_layers-1], None)[0]

            # First Layer
            if data[0] == None:
                target[0] = self.layers[0].activation(None,target[1],None, pre_top[0])[0]

            for i in range(2,self.num_layers-1,2):
                if data[i] == None:
                    target[i] = self.layers[i].activation(target[i-1], target[i+1], pre_bottom[i], pre_top[i])[0]

            #Top layer if odd
            if self.even_num_layers == False:
                if data[self.num_layers-1] == None:
                    target[self.num_layers-1] = self.layers[self.num_layers-1].activation(target[self.num_layers-2], None, pre_bottom[self.num_layers-1], None)[0]
        return target

    def sample(self, chains, k , target = None):
        ''' Samples all states k times.
            
        :Parameters:
            chains:     Markov chains.
                       -type: list of numpy arrays
                                  
            k:          Number of Gibbs sampling steps.
                       -type: int
                       
            target:     Resulting variable, if None inplace calculation is performed,
                       -type: int
            
        ''' 
        if target == None:
            target = []
            for i in range(0,self.num_layers):
                target.append(None)
    
        for i in range(1,self.num_layers-1,2):
            target[i] = self.layers[i].sample(self.layers[i].activation(chains[i-1], chains[i+1]))
        #Top layer if even
        if self.even_num_layers == True:
            target[self.num_layers-1] = self.layers[self.num_layers-1].sample(self.layers[self.num_layers-1].activation(chains[self.num_layers-2], None))
        # First Layer
        target[0] = self.layers[0].sample(self.layers[0].activation(None,target[1]))
        for i in range(2,self.num_layers-1,2):
            target[i] = self.layers[i].sample(self.layers[i].activation(target[i-1], target[i+1]))
        #Top layer if odd
        if self.even_num_layers == False:
            target[self.num_layers-1] = self.layers[self.num_layers-1].sample(self.layers[self.num_layers-1].activation(target[self.num_layers-2], None))
        for i in range(k-1):
            for i in range(1,self.num_layers-1,2):
                target[i] = self.layers[i].sample(self.layers[i].activation(target[i-1], target[i+1]))
            #Top layer if even
            if self.even_num_layers == True:
                target[self.num_layers-1] = self.layers[self.num_layers-1].sample(self.layers[self.num_layers-1].activation(target[self.num_layers-2], None))
            # First Layer
            target[0] = self.layers[0].sample(self.layers[0].activation(None,target[1]))
            for i in range(2,self.num_layers-1,2):
                target[i] = self.layers[i].sample(self.layers[i].activation(target[i-1], target[i+1]))
            #Top layer if odd
            if self.even_num_layers == False:
                target[self.num_layers-1] = self.layers[self.num_layers-1].sample(self.layers[self.num_layers-1].activation(target[self.num_layers-2], None))
        return target

