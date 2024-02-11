import numpy as numx

import pydeep.misc.io as IO
import pydeep.misc.visualization as STATISTICS
import pydeep.misc.visualization as VISUALIZATION
import pydeep.rbm.estimator as ESTIMATOR
import pydeep.rbm.model as MODEL
import pydeep.rbm.trainer as TRAINER

# Model Parameters
h1 = 20
h2 = 20
v1 = 10
v2 = 10
v1_org = 14
v2_org = 14

# Load and whiten data
data = numx.random.permutation(IO.load_matlab_file('../../../workspacePy/data/NaturalImage.mat','rawImages'))
data = preprocessing.remove_rows_means(data)
pca = preprocessing.PCA(v1_org * v2_org, True)
pca.train(data)
data = pca.project(data,v1*v2)

'''
# Training paramters
batch_size = 100
epochs = 200
k = 1
eps = [0.01,0.0,0.01,0.001]
mom = 0.9
decay = 0.0
max_norm = 0.01*numx.max(npExt.get_norms(data, axis = 1))

# Create trainer and model
grbm = MODEL.GaussianBinaryVarianceRBM(number_visibles = v1*v2, 
                                      number_hiddens = h1*h2, 
                                      data=data, 
                                      initial_weights=0.01, 
                                      initial_visible_bias=0.0, 
                                      initial_hidden_bias=-4.0, 
                                      initial_visible_offsets=0.0, 
                                      initial_hidden_offsets=0.0)

# Create trainer
gtrainer = TRAINER.CD(grbm)
st = True
# Train model
for epoch in range(0,epochs) :
    if epoch == 50:
        st = False
    if epoch == 100:
        eps[3] = 0.0
    if epoch == 150:
        eps[0] = 0.0
    for b in range(0,data.shape[0],batch_size):
        batch = data[b:b+batch_size,:]    
        gtrainer.train_images(data = batch,
                      num_epochs=1, 
                      epsilon=eps, 
                      k=k, 
                      momentum=mom,  
                      desired_sparseness=None, 
                      update_visible_offsets=0.0, 
                      update_hidden_offsets=0.0, 
                      restrict_gradient=max_norm, 
                      restriction_norm='Cols', 
                      use_hidden_states=st,
                      use_centered_gradient = False)
    print numx.mean(ESTIMATOR.reconstruction_error(grbm, data)) 


VISUALIZATION.imshow_matrix(VISUALIZATION.tile_matrix_rows(pca.unproject(grbm.w.T).T, v1_org,v2_org, h1, h2, border_size = 1,normalized = True), 'Weights')
samples = STATISTICS.generate_samples(grbm, data[0:30], 30, 1, v1_org, v2_org, False, pca)
VISUALIZATION.imshow_matrix(samples,'Samples')

IO.save_object(grbm, 'g.rbm')

'''
grbm = IO.load_object('g.rbm')

batch_size = 100
epochs = 20
k = 1
eps = [0.1,0.1,0.1]
mom = 0.9
decay = 0.0

data2 = grbm.probability_h_given_v(data)

v12 = h1
v22 = h2
h12 = 8
h22 = 8

brbm = MODEL.BinaryBinaryRBM(number_visibles = v12*v22, 
                             number_hiddens = h12*h22, 
                             data=data2, 
                             initial_weights='AUTO', 
                             initial_visible_bias='AUTO', 
                             initial_hidden_bias='AUTO', 
                             initial_visible_offsets='AUTO', 
                             initial_hidden_offsets='AUTO')

# Create trainer
btrainer = TRAINER.CD(brbm,data2)

# Train model
for epoch in range(0,epochs) :
    for b in range(0,data2.shape[0],batch_size):
        batch = data2[b:b+batch_size,:]    
        btrainer.train(data = batch,
                       num_epochs=1, 
                       epsilon=eps, 
                       k=k, 
                       momentum=mom,  
                       desired_sparseness=None, 
                       regSparseness=None, 
                       update_visible_offsets=0.01, 
                       update_hidden_offsets=0.01, 
                       restrict_gradient=None, 
                       restriction_norm='Cols', 
                       use_hidden_states=False,
                       use_centered_gradient = False)
    print numx.mean(ESTIMATOR.reconstruction_error(brbm, data2)) 
    
VISUALIZATION.imshow_matrix(VISUALIZATION.tile_matrix_rows(pca.unproject(numx.dot(grbm.w,brbm.w).T).T, v1_org,v2_org, h12, h22, border_size = 1,normalized = True), 'Weights 1')
VISUALIZATION.imshow_matrix(VISUALIZATION.tile_matrix_rows(brbm.w, h1,h2, h22, h22, border_size = 1,normalized = True), 'Weights 2')

samples = STATISTICS.generate_samples(brbm, data2[0:30], 30, 1, h1, h2, False, None)
VISUALIZATION.imshow_matrix(samples,'Samples')

VISUALIZATION.show()




