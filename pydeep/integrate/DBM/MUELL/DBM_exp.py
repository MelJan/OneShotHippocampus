'''
# ----------------------------------------------------
# Training a Centered Deep Boltzmann Machine
# ----------------------------------------------------
#
# Copyright: Gregoire Montavon
#
# This code is released under the MIT licence:
# http://www.opensource.org/licenses/mit-license.html
#
# ----------------------------------------------------
#
# This code is based on the paper:
#
#   G. Montavon, K.-R. Mueller
#   Deep Boltzmann Machines and the Centering Trick
#   in Neural Networks: Tricks of the Trade, 2nd Edn
#   Springer LNCS, 2012
#
# ----------------------------------------------------
#
# This code is a basic implementation of the centered
# deep Boltzmann machines (without model averaging,
# minibatching and other optimization hacks). The code
# also requires the MNIST dataset that can be
# downloaded at http://yann.lecun.com/exdb/mnist/.
#
# ----------------------------------------------------

import numpy,Image
import pydeep.misc.io as IO

# ====================================================
# Global parameters
# ====================================================
lr      = 0.05     # learning rate
rr      = 1     # reparameterization rate
mb      = 25        # minibatch size
hlayers = [100,25] # size of hidden layers
biases  = [-1,-1]   # initial biases on hidden layers

# ====================================================
# Helper functions
# ====================================================
def arcsigm(x): return numpy.arctanh(2*x-1)*2
def sigm(x):    return (numpy.tanh(x/2)+1)/2
def realize(x): return (x > numpy.random.uniform(0,1,x.shape))*1.0
def render(x,name):
    x = x - x.min() + 1e-9
    x = x / (x.max() + 1e-9)
    Image.fromarray((x*255).astype('byte'),'L').save(name)

# ====================================================
# Centered deep Boltzmann machine
# ----------------------------------------------------
# - self.W: list of weight matrices between layers
# - self.B: list of bias associated to each unit
# - self.O: list of offsets associated to each unit
# - self.X: free particles tracking model statistics
# ====================================================
class DBM:
    # --------------------------------------------
    # Initialize model parameters and particles
    # --------------------------------------------
    def __init__(self,M,B):
        self.W = [numpy.zeros([m,n]).astype('float32') for m,n in zip(M[:-1],M[1:])]
        self.B = [numpy.zeros([m]).astype('float32')+b for m,b in zip(M,B)]
        self.O = [sigm(b) for b in self.B]
        self.X = [numpy.zeros([mb,m]).astype('float32')+o for m,o in zip(M,self.O)]

    # --------------------------------------------
    # Gibbs activation of a layer
    # --------------------------------------------
    def gibbs(self,X,l):
        bu = numpy.dot(X[l-1]-self.O[l-1],self.W[l-1]) if l   > 0      else 0
        td = numpy.dot(X[l+1]-self.O[l+1],self.W[l].T) if l+1 < len(X) else 0
        X[l] = realize(sigm(bu+td+self.B[l]))

    # --------------------------------------------
    # Reparameterization
    # --------------------------------------------
    def reparamB(self,X,i):
        bu = numpy.dot((X[i-1]-self.O[i-1]),self.W[i-1]).mean(axis=0) if i   > 0      else 0
        td = numpy.dot((X[i+1]-self.O[i+1]),self.W[i].T).mean(axis=0) if i+1 < len(X) else 0
        self.B[i] = (1-rr)*self.B[i] + rr*(self.B[i] + bu + td)

    def reparamO(self,X,i):
        self.O[i] = (1-rr)*self.O[i] + rr*X[i].mean(axis=0)

    # --------------------------------------------
    # Learning step
    # --------------------------------------------
    def learn(self,Xd):

        # Initialize a data particle
        X = [realize(Xd)]+[self.X[l]*0+self.O[l] for l in range(1,len(self.X))]
        
        # Alternate gibbs sampler on data and free particles
        for l in (range(1,len(self.X),2)+range(2,len(self.X),2))*5: self.gibbs(X,l)
        for l in (range(1,len(self.X),2)+range(0,len(self.X),2))*1: self.gibbs(self.X,l)
        
        # Parameter update
        for i in range(0,len(self.W)):
            self.W[i] += lr*(numpy.dot((     X[i]-self.O[i]).T,     X[i+1]-self.O[i+1]) -
                             numpy.dot((self.X[i]-self.O[i]).T,self.X[i+1]-self.O[i+1]))/len(Xd)
            if i == 0:
                self.W[i] -= (lr * 0.001* numpy.sign(self.W[i])) 
             
        for i in range(0,len(self.B)):
            self.B[i] += lr*(X[i]-self.X[i]).mean(axis=0)
        
        #self.B[1] += lr*(0.01-self.X[1].mean(axis=0))
        #print numpy.mean(self.X[1].mean(axis=0))
        # Reparameterization
        for l in range(0,len(self.B)): self.reparamB(X,l)
        for l in range(0,len(self.O)): self.reparamO(X,l)

# ====================================================
# Example of execution
# ====================================================

# Initialize MNIST dataset and centered DBM
X = IO.load_MNIST("../../../workspacePy/data/mnist.pkl.gz",True)[0]
nn = DBM([784]+hlayers,[arcsigm(numpy.clip(X.mean(axis=0),0.01,0.99))]+biases)

for it in range(1000):

    # Perform some learning steps
    for _ in range(2000): nn.learn(X[:mb])
    
    # Output some debugging information
    print(("%03d |" + " %.3f "*len(nn.W))%tuple([it]+[W.std() for W in nn.W]))
    W = 1
    for l in range(len(nn.W)):
        W = numpy.dot(W,nn.W[l])
        m = int(W.shape[1]**.5)
        render(W.reshape([28,28,m,m]).transpose([2,0,3,1]).reshape([28*m,28*m]),'W%d.jpg'%(l+1));
    render((nn.X[0]).reshape([mb,28,28]).transpose([1,0,2]).reshape([28,mb*28]),'X.jpg');
    print numpy.sum(numpy.mean((nn.X[0]-X[0])**2.0,axis=1))/25
    import pydeep.base.numpyextension as numxExt
    print numxExt.get_norms(nn.B[0])

'''

'''
# ----------------------------------------------------
# Training a Centered Deep Boltzmann Machine
# ----------------------------------------------------
#
# Copyright: Gregoire Montavon
#
# This code is released under the MIT licence:
# http://www.opensource.org/licenses/mit-license.html
#
# ----------------------------------------------------
#
# This code is based on the paper:
#
#   G. Montavon, K.-R. Mueller
#   Deep Boltzmann Machines and the Centering Trick
#   in Neural Networks: Tricks of the Trade, 2nd Edn
#   Springer LNCS, 2012
#
# ----------------------------------------------------
#
# This code is a basic implementation of the centered
# deep Boltzmann machines (without model averaging,
# minibatching and other optimization hacks). The code
# also requires the MNIST dataset that can be
# downloaded at http://yann.lecun.com/exdb/mnist/.
#
# ----------------------------------------------------

import numpy,Image
import pydeep.misc.io as IO

# ====================================================
# Global parameters
# ====================================================
lr      = 0.05     # learning rate
rr      = 1     # reparameterization rate
mb      = 25        # minibatch size
hlayers = [100,25] # size of hidden layers
biases  = [-1,-1]   # initial biases on hidden layers

# ====================================================
# Helper functions
# ====================================================
def arcsigm(x): return numpy.arctanh(2*x-1)*2
def sigm(x):    return (numpy.tanh(x/2)+1)/2
def realize(x): return (x > numpy.random.uniform(0,1,x.shape))*1.0
def render(x,name):
    x = x - x.min() + 1e-9
    x = x / (x.max() + 1e-9)
    Image.fromarray((x*255).astype('byte'),'L').save(name)

# ====================================================
# Centered deep Boltzmann machine
# ----------------------------------------------------
# - self.W: list of weight matrices between layers
# - self.B: list of bias associated to each unit
# - self.O: list of offsets associated to each unit
# - self.X: free particles tracking model statistics
# ====================================================
class DBM:
    # --------------------------------------------
    # Initialize model parameters and particles
    # --------------------------------------------
    def __init__(self,M,B):
        self.W = [numpy.zeros([m,n]).astype('float32') for m,n in zip(M[:-1],M[1:])]
        self.B = [numpy.zeros([m]).astype('float32')+b for m,b in zip(M,B)]
        self.O = [sigm(b) for b in self.B]
        self.X = [numpy.zeros([mb,m]).astype('float32')+o for m,o in zip(M,self.O)]

    # --------------------------------------------
    # Gibbs activation of a layer
    # --------------------------------------------
    def gibbs(self,X,l):
        bu = numpy.dot(X[l-1]-self.O[l-1],self.W[l-1]) if l   > 0      else 0
        td = numpy.dot(X[l+1]-self.O[l+1],self.W[l].T) if l+1 < len(X) else 0
        X[l] = realize(sigm(bu+td+self.B[l]))

    # --------------------------------------------
    # Reparameterization
    # --------------------------------------------
    def reparamB(self,X,i):
        bu = numpy.dot((X[i-1]-self.O[i-1]),self.W[i-1]).mean(axis=0) if i   > 0      else 0
        td = numpy.dot((X[i+1]-self.O[i+1]),self.W[i].T).mean(axis=0) if i+1 < len(X) else 0
        self.B[i] = (1-rr)*self.B[i] + rr*(self.B[i] + bu + td)

    def reparamO(self,X,i):
        self.O[i] = (1-rr)*self.O[i] + rr*X[i].mean(axis=0)

    # --------------------------------------------
    # Learning step
    # --------------------------------------------
    def learn(self,Xd):

        # Initialize a data particle
        X = [realize(Xd)]+[self.X[l]*0+self.O[l] for l in range(1,len(self.X))]
        
        # Alternate gibbs sampler on data and free particles
        for l in (range(1,len(self.X),2)+range(2,len(self.X),2))*5: self.gibbs(X,l)
        for l in (range(1,len(self.X),2)+range(0,len(self.X),2))*1: self.gibbs(self.X,l)
        
        # Parameter update
        for i in range(0,len(self.W)):
            self.W[i] += lr*(numpy.dot((     X[i]-self.O[i]).T,     X[i+1]-self.O[i+1]) -
                             numpy.dot((self.X[i]-self.O[i]).T,self.X[i+1]-self.O[i+1]))/len(Xd)
            if i == 0:
                self.W[i] -= (lr * 0.001* numpy.sign(self.W[i])) 
             
        for i in range(0,len(self.B)):
            self.B[i] += lr*(X[i]-self.X[i]).mean(axis=0)
        
        #self.B[1] += lr*(0.01-self.X[1].mean(axis=0))
        #print numpy.mean(self.X[1].mean(axis=0))
        # Reparameterization
        for l in range(0,len(self.B)): self.reparamB(X,l)
        for l in range(0,len(self.O)): self.reparamO(X,l)

# ====================================================
# Example of execution
# ====================================================

# Initialize MNIST dataset and centered DBM
X = IO.load_MNIST("../../../workspacePy/data/mnist.pkl.gz",True)[0]
nn = DBM([784]+hlayers,[arcsigm(numpy.clip(X.mean(axis=0),0.01,0.99))]+biases)

for it in range(1000):

    # Perform some learning steps
    for _ in range(2000): nn.learn(X[:mb])
    
    # Output some debugging information
    print(("%03d |" + " %.3f "*len(nn.W))%tuple([it]+[W.std() for W in nn.W]))
    W = 1
    for l in range(len(nn.W)):
        W = numpy.dot(W,nn.W[l])
        m = int(W.shape[1]**.5)
        render(W.reshape([28,28,m,m]).transpose([2,0,3,1]).reshape([28*m,28*m]),'W%d.jpg'%(l+1));
    render((nn.X[0]).reshape([mb,28,28]).transpose([1,0,2]).reshape([28,mb*28]),'X.jpg');
    print numpy.sum(numpy.mean((nn.X[0]-X[0])**2.0,axis=1))/25
    import pydeep.base.numpyextension as numxExt
    print numxExt.get_norms(nn.B[0])
'''

import numpy as numx

import pydeep.base.numpyextension as numxExt
import pydeep.misc.io as IO
import pydeep.misc.visualization as VIS
from pydeep.base.activationfunction import Sigmoid


def generate_2D_connection_matrix(    input_x_dim, 
                                          input_y_dim, 
                                          field_x_dim, 
                                          field_y_dim, 
                                          overlap_x_dim, 
                                          overlap_y_dim, 
                                          wrap_around = True):
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

# Load and whiten data

#data = numx.random.permutation(IO.load_matlab_file('../../../workspacePy/data/NaturalImage.mat','rawImages'))
#data = PRE.remove_rows_means(data)
#zca = PRE.ZCA(14*14)
#zca.train_images(data)
#data = zca.project(data)

# Load data and whiten it
data = IO.load_MNIST("../../../workspacePy/data/mnist.pkl.gz",False)[0]
conn1 = generate_2D_connection_matrix(input_x_dim = 28, 
                                     input_y_dim = 28, 
                                     field_x_dim = 14, 
                                     field_y_dim = 14, 
                                     overlap_x_dim = 12, 
                                     overlap_y_dim = 12, 
                                     wrap_around = True)
conn2 = generate_2D_connection_matrix(input_x_dim = 14, 
                                     input_y_dim = 14, 
                                     field_x_dim = 2, 
                                     field_y_dim = 2, 
                                     overlap_x_dim = 1, 
                                     overlap_y_dim = 1, 
                                     wrap_around = True)

#data = PROBLEM.generate_bars_and_stripes_complete(4)
#data = numx.vstack((data[0],data,data[5]))
v1 = v2 = 28
h1 = h2 = 14
z1 = z2 = 14
N = v1*v2
M = h1*h2
O = z1*z2

batch_size = 25
epochs = 5
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

#lr_s1 = 0.0



W1 = numx.random.randn(N, M) * 0.001
W1 *= conn1
W2 = numx.random.randn(M, O) * 0.001
W2 *= conn2

#s1 = numx.zeros((1,N)) + numx.std(numx.clip(data,0.001,numx.finfo(numx.float64).max))

o1 = numx.mean(data, axis = 0).reshape(1,N)
o2 = numx.zeros((1,M)) + 0.5
o3 = numx.zeros((1,O)) + 0.5

b1 = Sigmoid.g(numx.clip(o1,0.01,0.99))
#b1 = numx.zeros((1,N))
b2 = numx.zeros((1,M))
b3 = numx.zeros((1,O))

m1 = o1+numx.zeros((batch_size,N))
m2 = o2+numx.zeros((batch_size,M))
m3 = o3+numx.zeros((batch_size,O))

for epoch in range(0,epochs) :
    err = 0.0
    for b in range(0,data.shape[0],batch_size):
        d1 = data[b:b+batch_size,:]
        id1 = numx.dot(d1-o1,W1)
        d3 = o3
        for _ in range(k_pos):  
            d2 = Sigmoid.f( id1 + numx.dot(d3-o3,W2.T) + b2)
            d2 = numx.double(d2 > numx.random.random(d2.shape))
            d3 = Sigmoid.f(numx.dot(d2-o2,W2) + b3)
            d3 = numx.double(d3 > numx.random.random(d3.shape))
        for _ in range(k_neg):  
            m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(m3-o3,W2.T) + b2)
            #m2 = Sigmoid.f(numx.dot((m1-o1)/(s1**2),W1) + numx.dot(m3-o3,W2.T) + b2)
            m2 = numx.double(m2 > numx.random.random(m2.shape))
            m1 = Sigmoid.f(numx.dot(m2-o2,W1.T) + b1)
            m1 = numx.double(m1 > numx.random.random(m1.shape))
            m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
            m3 = numx.double(m3 > numx.random.random(m3.shape))

            #m1 = numx.dot((m2-o2),W2.T) + b1 + o1
            #m1 = m1 + numx.random.randn(m1.shape[0], m1.shape[1])*s1
        #m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(m3-o3,W2.T) + b2)
        #m2 = Sigmoid.f(numx.dot((m1-o1)/(s1**2),W1) + numx.dot(m3-o3,W2.T) + b2)
        #m2 = m2 > numx.random.random(m2.shape)
        #m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
        #m3 = m3 > numx.random.random(m3.shape)

        new_o1 = d1.mean(axis=0)
        new_o2 = d2.mean(axis=0)
        new_o3 = d3.mean(axis=0)
        
        # Reparameterize
        b1 += lr_o2*numx.dot(new_o2-o2,W1.T)
        b2 += lr_o1*numx.dot(new_o1-o1,W1) + lr_o3*numx.dot(new_o3-o3,W2.T)
        b3 += lr_o2*numx.dot(new_o2-o2,W2)
        
        o1 = (1.0-lr_o1)*o1 + lr_o1*new_o1
        o2 = (1.0-lr_o2)*o2 + lr_o2*new_o2
        o3 = (1.0-lr_o3)*o3 + lr_o3*new_o3

        W1 += lr_W1/batch_size*(numx.dot((d1-o1).T,d2-o2)-numx.dot((m1-o1).T,m2-o2))
        W2 += lr_W2/batch_size*(numx.dot((d2-o2).T,d3-o3)-numx.dot((m2-o2).T,m3-o3))
        if epoch < 2:
            W1 *= conn1
        W2 *= conn2
        #s1 += lr_s1/batch_size*(((m1-b1-o1)**2 - 2.0 * (m1 - o1) * numx.dot(m2, W1.T)).sum(axis=0) / (s1**3)).reshape(1,N)
        b1 += lr_b1/batch_size*(numx.sum(d1-m1,axis = 0)).reshape(1,N)
        b2 += lr_b2/batch_size*(numx.sum(d2-m2,axis = 0)).reshape(1,M)
        b3 += lr_b3/batch_size*(numx.sum(d3-m3,axis = 0)).reshape(1,O)
        
        err+=numx.sum(numx.mean((d1-m1)**2.0,axis=1))
    print err/data.shape[0]
    print numx.mean(numxExt.get_norms(W1))
    print numxExt.get_norms(b2, axis = 1)
VIS.imshow_matrix(VIS.tile_matrix_rows(W1, v1,v2, h1, h2, border_size = 1,normalized = False), 'Weights 1')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(W1,W2), v1,v2, z2, z2, border_size = 1,normalized = False), 'Weights 2')

VIS.imshow_matrix(VIS.tile_matrix_columns(d1, v1, v2, 5, 5, 1, False),'d')
m2 = Sigmoid.f(numx.dot(d1-o1,W1) + numx.dot(d3-o3,W2.T) + b2)
m1 = Sigmoid.f(numx.dot(m2-o2,W1.T) + b1)
m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v1, v2, 5, 5, 1, False),'m 1')
m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(d3-o3,W2.T) + b2)
m1 = Sigmoid.f(numx.dot(m2-o2,W1.T) + b1)
m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v1, v2, 5, 5, 1, False),'m 2')
m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(d3-o3,W2.T) + b2)
m1 = Sigmoid.f(numx.dot(m2-o2,W1.T) + b1)
m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
VIS.imshow_matrix(VIS.tile_matrix_columns(m1, v1, v2, 5, 5, 1, False),'m 3')
m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(d3-o3,W2.T) + b2)
m1 = Sigmoid.f(numx.dot(m2-o2,W1.T) + b1)
m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)

VIS.imshow_matrix(W1, 'Weights r1')
VIS.imshow_matrix(W2, 'Weights r2')
VIS.imshow_matrix(conn1, 'Conn r1')
VIS.imshow_matrix(conn2, 'Conn r2')
VIS.show()

'''

import numpy as numx
import pydeep.base.numpyextension as numxExt
from pydeep.base.activationfunction import Sigmoid
import pydeep.preprocessing as PRE
import pydeep.misc.visualization as VIS
import pydeep.misc.io as IO

# Load and whiten data
"""
data = numx.random.permutation(IO.load_matlab_file('../../../workspacePy/data/NaturalImage.mat','rawImages'))
data = PRE.remove_rows_means(data)
zca = PRE.ZCA(14*14)
zca.train_images(data)
data = zca.project(data)
"""

# Load data and whiten it
data = IO.load_MNIST("../../../workspacePy/data/mnist.pkl.gz",True)[0]
print range(1,3,2)+range(2,3,2)
print range(1,3,2)+range(0,3,2)
v1 = v2 = 28
h1 = h2 = 10
z1 = z2 = 5
N = v1*v2
M = h1*h2
O = z1*z2

batch_size = 25
epochs = 10
k = 1

lr_W1 = 0.005
lr_W2 = 0.005

lr_b1 = 0.005
lr_b2 = 0.005
lr_b3 = 0.005

lr_o1 = 0.001
lr_o2 = 0.001
lr_o3 = 0.001

#lr_s1 = 0.0



W1 = numx.random.randn(N, M) * 0.01
W2 = numx.random.randn(M, O) * 0.01

#s1 = numx.zeros((1,N)) + numx.std(numx.clip(data,0.001,numx.finfo(numx.float64).max))

o1 = numx.mean(data, axis = 0).reshape(1,N)
o2 = numx.zeros((1,M)) + 0.0
o3 = numx.zeros((1,O)) + 0.0

b1 = Sigmoid.g(numx.clip(o1,0.01,0.99))
#b1 = numx.zeros((1,N))
b2 = numx.zeros((1,M))
b3 = numx.zeros((1,O))

m1 = o1*numx.zeros((batch_size,N))
m2 = o2*numx.zeros((batch_size,M))
m3 = o3*numx.zeros((batch_size,O))

for epoch in range(0,epochs) :
    err = 0.0
    for b in range(0,data.shape[0],batch_size):
        d1 = data[b:b+batch_size,:]
        id1 = numx.dot(d1-o1,W1)
        d3 = o3
        for _ in range(5):  
            d2 = Sigmoid.f( id1 + numx.dot(d3-o3,W2.T) + b2)
            d2 = d2 > numx.random.random(d2.shape)
            d3 = Sigmoid.f(numx.dot(d2-o2,W2) + b3)
            d3 = d3 > numx.random.random(d3.shape)
        for _ in range(k):  
            m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(m3-o3,W2.T) + b2)
            #m2 = Sigmoid.f(numx.dot((m1-o1)/(s1**2),W1) + numx.dot(m3-o3,W2.T) + b2)
            m2 = m2 > numx.random.random(m2.shape)
            m1 = Sigmoid.f(numx.dot(m2-o2,W1.T) + b1)
            m1 = m1 > numx.random.random(m1.shape)
            m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
            m3 = m3 > numx.random.random(m3.shape)

            #m1 = numx.dot((m2-o2),W2.T) + b1 + o1
            #m1 = m1 + numx.random.randn(m1.shape[0], m1.shape[1])*s1
        #m2 = Sigmoid.f(numx.dot(m1-o1,W1) + numx.dot(m3-o3,W2.T) + b2)
        #m2 = Sigmoid.f(numx.dot((m1-o1)/(s1**2),W1) + numx.dot(m3-o3,W2.T) + b2)
        #m2 = m2 > numx.random.random(m2.shape)
        #m3 = Sigmoid.f(numx.dot(m2-o2,W2) + b3)
        #m3 = m3 > numx.random.random(m3.shape)

        new_o1 = d1.mean(axis=0)
        new_o2 = d2.mean(axis=0)
        new_o3 = d3.mean(axis=0)
        
        # Reparameterize
        b1 += lr_o2*numx.dot(new_o2-o2,W1.T)
        b2 += lr_o1*numx.dot(new_o1-o1,W1) + lr_o3*numx.dot(new_o3-o3,W2.T)
        b3 += lr_o2*numx.dot(new_o2-o2,W2)
        
        o1 = (1.0-lr_o1)*o1 + lr_o1*new_o1
        o2 = (1.0-lr_o2)*o2 + lr_o2*new_o2
        o3 = (1.0-lr_o3)*o3 + lr_o3*new_o3

        W1 += lr_W1/batch_size*(numx.dot((d1-o1).T,d2-o2)-numx.dot((m1-o1).T,m2-o2))
        W2 += lr_W2/batch_size*(numx.dot((d2-o2).T,d3-o3)-numx.dot((m2-o2).T,m3-o3))
        #s1 += lr_s1/batch_size*(((m1-b1-o1)**2 - 2.0 * (m1 - o1) * numx.dot(m2, W1.T)).sum(axis=0) / (s1**3)).reshape(1,N)
        b1 += lr_b1/batch_size*(numx.sum(d1-m1,axis = 0)).reshape(1,N)
        b2 += lr_b2/batch_size*(numx.sum(d2-m2,axis = 0)).reshape(1,M)
        b3 += lr_b3/batch_size*(numx.sum(d3-m3,axis = 0)).reshape(1,O)
        
        err+=numx.sum(numx.mean((d1-m1)**2.0,axis=1))
    print err/data.shape[0]
    print numx.mean(numxExt.get_norms(W1))
    print numxExt.get_norms(b2, axis = 1)
VIS.imshow_matrix(VIS.tile_matrix_rows(W1, v1,v2, h1, h2, border_size = 1,normalized = True), 'Weights')
VIS.imshow_matrix(VIS.tile_matrix_rows(numx.dot(W1,W2), v1,v2, z2, z2, border_size = 1,normalized = True), 'Weights')
VIS.show()
'''