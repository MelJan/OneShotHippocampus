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

    def meanfield(self, chains, k , fixed_layer = None, inplace = True):
        ''' Samples all states k times.
            
        :Parameters:
            chains:     Markov chains.
                       -type: list of numpy arrays
                                  
            k:          Number of Gibbs sampling steps.
                       -type: int
                       
            target:     Resulting variable, if None inplace calculation is performed,
                       -type: int
            
        ''' 
        if isinstance(fixed_layer,list):
            if len(fixed_layer) != len(chains):
                raise Exception("fixed_layer has to have the same length as chains!")
        else:
            fixed_layer = []
            for i in range(0,self.num_layers):
                fixed_layer.append(False)

        # Create empty list with pre caluclated values
        pre_top = []
        pre_bottom = []
        for i in range(self.num_layers):
            pre_top.append(None)
            pre_bottom.append(None)

        # Get all pre calculatable values
        for i in range(self.num_layers):
            if fixed_layer[i] is True:
                if i > 0:
                    pre_top[i-1] = self.layers[i].input_weight_layer.propagate_down(chains[i])
                if i < self.num_layers-1:
                    pre_bottom[i+1] = self.layers[i].output_weight_layer.propagate_up(chains[i])

        if inplace is True:
            target = chains
        else:
            target = []
            for i in range(0,self.num_layers):
                target.append(numx.copy(chains[i]))
    
        for i in range(k):
            for i in range(1,self.num_layers-1,2):
                if fixed_layer[i] is False:
                    target[i] = self.layers[i].activation(target[i-1], target[i+1], pre_bottom[i], pre_top[i])[0]
            #Top layer if even
            if self.even_num_layers is True:
                if fixed_layer[self.num_layers-1] is False:
                    target[self.num_layers-1] = self.layers[self.num_layers-1].activation(target[self.num_layers-2], None, pre_bottom[self.num_layers-1], None)[0]
            # First Layer
            if fixed_layer[0] is False:
                target[0] = self.layers[0].activation(None,target[1],None, pre_top[0])[0]
            for i in range(2,self.num_layers-1,2):
                if fixed_layer[i] is False:
                    target[i] = self.layers[i].activation(target[i-1], target[i+1], pre_bottom[i], pre_top[i])[0]
            #Top layer if odd
            if self.even_num_layers is False:
                if fixed_layer[self.num_layers-1] is False:
                    target[self.num_layers-1] = self.layers[self.num_layers-1].activation(target[self.num_layers-2], None, pre_bottom[self.num_layers-1], None)[0]
        return target
    
    def sample(self, chains, k , fixed_layer = None, inplace = True):
        ''' Samples all states k times.
            
        :Parameters:
            chains:     Markov chains.
                       -type: list of numpy arrays
                                  
            k:          Number of Gibbs sampling steps.
                       -type: int
                       
            target:     Resulting variable, if None inplace calculation is performed,
                       -type: int
            
        ''' 
        if isinstance(fixed_layer,list):
            if len(fixed_layer) != len(chains):
                raise Exception("fixed_layer has to have the same length as chains!")
        else:
            fixed_layer = []
            for i in range(0,self.num_layers):
                fixed_layer.append(False)

        # Create empty list with pre caluclated values
        pre_top = []
        pre_bottom = []
        for i in range(self.num_layers):
            pre_top.append(None)
            pre_bottom.append(None)

        # Get all pre calculatable values
        for i in range(self.num_layers):
            if fixed_layer[i] is True:
                if i > 0:
                    pre_top[i-1] = self.layers[i].input_weight_layer.propagate_down(chains[i])
                if i < self.num_layers-1:
                    pre_bottom[i+1] = self.layers[i].output_weight_layer.propagate_up(chains[i])

        if inplace is True:
            target = chains
        else:
            target = []
            for i in range(0,self.num_layers):
                target.append(numx.copy(chains[i]))
    
        for i in range(k):
            for i in range(1,self.num_layers-1,2):
                if fixed_layer[i] is False:
                    target[i] = self.layers[i].sample(self.layers[i].activation(target[i-1], target[i+1], pre_bottom[i], pre_top[i]))
            #Top layer if even
            if self.even_num_layers is True:
                if fixed_layer[self.num_layers-1] is False:
                    target[self.num_layers-1] = self.layers[self.num_layers-1].sample(self.layers[self.num_layers-1].activation(target[self.num_layers-2], None, pre_bottom[self.num_layers-1], None))
            # First Layer
            if fixed_layer[0] is False:
                target[0] = self.layers[0].sample(self.layers[0].activation(None,target[1],None, pre_top[0]))
            for i in range(2,self.num_layers-1,2):
                if fixed_layer[i] is False:
                    target[i] = self.layers[i].sample(self.layers[i].activation(target[i-1], target[i+1], pre_bottom[i], pre_top[i]))
            #Top layer if odd
            if self.even_num_layers is False:
                if fixed_layer[self.num_layers-1] is False:
                    target[self.num_layers-1] = self.layers[self.num_layers-1].sample(self.layers[self.num_layers-1].activation(target[self.num_layers-2], None,pre_bottom[self.num_layers-1], None))
        return target

    def update(self, chains_d, chains_m, lr_W, lr_b, lr_o ):
        
        for l,x_d in zip(self.layers, chains_d):
            l.update_offsets(numx.mean(x_d,axis = 0).reshape(1,x_d.shape[1]), lr_o)
        
        grad_w = []
        for l in range(1,self.num_layers):
            grad = self.layers[l].input_weight_layer.calculate_weight_gradients(chains_d[l-1],chains_d[l],chains_m[l-1],chains_m[l],self.layers[l-1].offset,self.layers[l].offset)
            self.layers[l].input_weight_layer.update_weights(lr_W*grad, None, None)
            grad_w.append(grad)
            
        for l in range(self.num_layers):
            if l == 0:
                grad_b =self.layers[0].calculate_gradient_b(chains_d[0], chains_m[0], None, self.layers[1].offset, None, grad_w[0])
            elif l == self.num_layers-1:
                grad_b =self.layers[l].calculate_gradient_b(chains_d[l], chains_m[l], self.layers[l-1].offset, None, grad_w[l-1], None)
            else:
                grad_b =self.layers[l].calculate_gradient_b(chains_d[l], chains_m[l], self.layers[l-1].offset,self.layers[l+1].offset, grad_w[l-1], grad_w[l])

            self.layers[l].update_biases(lr_b*grad_b, None, None)
