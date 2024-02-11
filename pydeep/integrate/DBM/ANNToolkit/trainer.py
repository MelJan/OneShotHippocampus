import numpy as numx
import pydeep.base.numpyextension as numxExt

def DBM_trainer(object):
    
    def __init__(self, model):
        ''' This function initializes the weight layer.
            
        :Parameters:
            model:      Model, basically a list of layers
                       -type: Weight_layer or None
                                  
            dtype:      Used data type i.e. numpy.float64
                       -type: numpy.float32 or numpy.float64 or 
                                           numpy.float128  
            
        '''  
        self.model = model
        model_data = []
        
    def train(self, data, lr_weights, lr_biases, lr_offsets, mf_conv = 0.0001):
        ''' This function initializes the weight layer.
            
        :Parameters:
            model:      Model, basically a list of layers
                       -type: Weight_layer or None
                                  
            dtype:      Used data type i.e. numpy.float64
                       -type: numpy.float32 or numpy.float64 or 
                                           numpy.float128  
            
        '''  
        # Initialize negative Markov chain
        x_m = numx.zeros((batch_size,v11*v12))+l1.offset
        y_m = numx.zeros((batch_size,v21*v22))+l2.offset
        z_m = numx.zeros((batch_size,v31*v32))+l3.offset
        a_m = numx.zeros((batch_size,v41*v42))+l4.offset

        # Reparameterize RBM such that the inital setting is the same for centereing and centered training
        l1.bias += numx.dot(0.0-l2.offset,wl1.weights.T)
        l2.bias += numx.dot(0.0-l1.offset,wl1.weights) + numx.dot(0.0-l3.offset,wl2.weights.T)
        l3.bias += numx.dot(0.0-l2.offset,wl2.weights) + numx.dot(0.0-l4.offset,wl3.weights.T)
        l4.bias += numx.dot(0.0-l3.offset,wl3.weights)


        d2_new = Sigmoid.f( id1 + numx.dot(d3-self.model.o3,self.model.W2.T) + self.model.b2)
        d3_new = Sigmoid.f(numx.dot(d2_new-self.model.o2,self.model.W2) + self.model.b3)
        while numx.max(numx.abs(d2_new-d2)) > meanfield or numx.max(numx.abs(d3_new-d3)) > meanfield: 
            d2 = d2_new
            d3 = d3_new
            d2_new = Sigmoid.f( id1 + numx.dot(d3_new-self.model.o3,self.model.W2.T) + self.model.b2)
            d3_new = Sigmoid.f(numx.dot(d2_new-self.model.o2,self.model.W2) + self.model.b3)
        d2 = d2_new
        d3 = d3_new


        #positive phase
        x_d = data[b:b+batch_size,:]
        y_d = numx.zeros((batch_size,v21*v22))+l2.offset
        z_d = numx.zeros((batch_size,v31*v32))+l3.offset
        a_d = numx.zeros((batch_size,v41*v42))+l4.offset
        for i in range(k_pos):
            y_d = l2.activation(x_d, z_d)[0]
            #y_d = l2.sample(y_d)
            a_d = l4.activation(z_d, None)[0]
            #a_d = l4.sample(a_d)
            z_d = l3.activation(y_d, a_d)[0]
            #z_d = l3.sample(z_d)
            
        #negative phase
        for i in range(k_neg):
            y_m = l2.activation(x_m, z_m)
            y_m = l2.sample(y_m)
            a_m = l4.activation(z_m, None)
            a_m = l4.sample(a_m)
            x_m = l1.activation(None,y_m)
            x_m = l1.sample(x_m)  
            z_m = l3.activation(y_m, a_m)
            z_m = l3.sample(z_m) 

        # Estimate new means and update
        l1.update_offsets(numx.mean(x_d,axis = 0).reshape(1,x_d.shape[1]), lr_o1)
        l2.update_offsets(numx.mean(y_d,axis = 0).reshape(1,y_d.shape[1]), lr_o2)
        l3.update_offsets(numx.mean(z_d,axis = 0).reshape(1,z_d.shape[1]), lr_o3)
        l4.update_offsets(numx.mean(a_d,axis = 0).reshape(1,a_d.shape[1]), lr_o4)
        
        # Calculate centered weight gradients and update
        wl1_grad = wl1.calculate_weight_gradients(x_d, y_d, x_m, y_m, l1.offset, l2.offset)
        wl2_grad = wl2.calculate_weight_gradients(y_d, z_d, y_m, z_m, l2.offset, l3.offset)
        wl3_grad = wl3.calculate_weight_gradients(z_d, a_d, z_m, a_m, l3.offset, l4.offset)
        wl1.update_weights(lr_W1*wl1_grad, None, None)
        wl2.update_weights(lr_W2*wl2_grad, None, None)
        wl3.update_weights(lr_W3*wl3_grad, None, None)
        
        # Calsulate centered gradients for biases and update
        grad_b1 =l1.calculate_gradient_b(x_d, x_m, None, l2.offset, None, wl1_grad)
        grad_b2 =l2.calculate_gradient_b(y_d, y_m, l1.offset, l3.offset, wl1_grad, wl2_grad)
        grad_b3 =l3.calculate_gradient_b(z_d, z_m, l2.offset, l4.offset, wl2_grad, wl3_grad)
        grad_b4 =l4.calculate_gradient_b(a_d, a_m, l3.offset, None, wl3_grad, None)

        #Einbauen letzten zustande saven in Unit layer
        l1.update_biases(lr_b1*grad_b1)
        l2.update_biases(lr_b2*grad_b2)
        l3.update_biases(lr_b3*grad_b3)
        l4.update_biases(lr_b4*grad_b4)
    