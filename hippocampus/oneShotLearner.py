import numpy as numx
import pydeep.misc.visualization as VIS
import pydeep.base.numpyextension as npExt
import pydeep.misc.io as IO
import pydeep.base.activationfunction as AFCT

class One_shot_learner(object):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 activation_function,
                 max_activity):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.act = activation_function
        self.max_activity = max_activity
        self.act_i = AFCT.Rectifier
        self.act_o = AFCT.Rectifier
        self.threshold = 0.1**2
        self.learning_rate = 1.0/self.input_dim
        self.noise_factor = 0.0
        self.confident = 0.1
        self.current_index = 0

        self.weights_in = numx.zeros((self.input_dim,self.hidden_dim))-self.confident
        self.weights_out = numx.zeros((self.hidden_dim,self.output_dim))-self.confident

    def _set_current_index(self):
        self.current_index=(self.current_index+1) % self.hidden_dim

    def _get_optimal_weights(self, x, y):
        # Generalized Hebb
        self.weights_in[:, self.current_index] = numx.zeros(self.input_dim)-self.confident
        diff = self.max_activity
        while numx.abs(diff) > self.threshold :
            diff = self.act_i.f(numx.dot(x, self.weights_in[:, self.current_index])+numx.random.randn()*self.noise_factor) - self.max_activity
            self.weights_in[:, self.current_index] -= (self.learning_rate*diff*x).reshape(784)

        self.weights_out[self.current_index,:] = numx.zeros(self.output_dim)-self.confident
        diff = self.max_activity
        while numx.sum(numx.abs(diff)) > self.threshold :
            diff = self.act_o.f(self.weights_out[self.current_index,:]+numx.random.randn()*self.noise_factor) - y
            self.weights_out[self.current_index,:] -= (self.learning_rate*diff).reshape(784)
        return self.weights_in[:, self.current_index],self.weights_out[self.current_index,:]

    def store_pattern(self,x,y):
        w1,w2 = self._get_optimal_weights(x,y)
        self.weights_in[:, self.current_index] = w1
        self.weights_out[self.current_index,:] = w2
        self._set_current_index()

    def calculate_output(self, h):
        return self.act_o.f(numx.dot(h,self.weights_out))

    def calculate_hidden(self, x):
        return self.act_i.f(numx.dot(x,self.weights_in))

    def calculate_input(self, o):
        return numx.dot(o,(self.weights_in/self._weight_factor).T)


s = 50
h = 10
data = IO.load_mnist("../../../data/mnist.pkl.gz",True)[0][0:s,:]

model = One_shot_learner(784,h,784,AFCT.Rectifier,1.0)
for i in range(data.shape[0]):
    print i
    model.store_pattern(data[i,:], data[(i+1)%s,:])


res_d =  VIS.tile_matrix_rows(data.T, 28, 28, s,1,0,False)
VIS.imshow_matrix(res_d,'data')

weights = model.weights_in
VIS.imshow_matrix(VIS.tile_matrix_rows(weights, 28, 28, h,1,0,False),'weights')

output =  model.calculate_hidden(data)
VIS.imshow_matrix(VIS.tile_matrix_rows(output.T, 1, h, s,1,0,False),'output')

res =  model.calculate_output(output)
VIS.imshow_matrix(VIS.tile_matrix_rows(res.T, 28, 28, s,1,0,False),'recon')

out = model.calculate_hidden(data)
output_thres = out >=numx.atleast_2d(numx.max(out,axis = 1)).T
VIS.imshow_matrix(VIS.tile_matrix_rows(output_thres.T, 1, h, s,1,0,False),'output max')

input_thres =  model.calculate_output(output_thres)
VIS.imshow_matrix(VIS.tile_matrix_rows(input_thres.T, 28, 28, s,1,0,False),'recon thres')

import pydeep.base.corruptor as Corr
kmax = Corr.KWinnerTakesAll(k = 8,axis = 0)
maxi = kmax.corrupt(out)
print maxi.shape
VIS.imshow_matrix(VIS.tile_matrix_rows(maxi.T, 1, h, s,1,0,False),'output 2')

VIS.show()

