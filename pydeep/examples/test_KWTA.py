
import numpy as numx

import pydeep.fnn.model as MODEL
import pydeep.fnn.layer as LAYER
import pydeep.fnn.trainer as TRAINER
import pydeep.base.activationfunction as ACT
import pydeep.base.costfunction as COST
import pydeep.base.corruptor as CORR
import pydeep.misc.io as IO
import pydeep.base.numpyextension as npExt

import mkl
mkl.set_num_threads(1)


# Set random seed (optional)
numx.random.seed(42)

# Load data and whiten it
train_data,train_label,valid_data, valid_label,test_data, test_label = IO.load_mnist("../../../data/mnist.pkl.gz",False)
train_data = numx.vstack((train_data,valid_data))
train_label = numx.hstack((train_label,valid_label)).T
train_label = npExt.get_binary_label(train_label)
test_label = npExt.get_binary_label(test_label)

# Create model
l1 = LAYER.FullConnLayer(input_dim = train_data.shape[1],
                         output_dim = 100,
                         activation_function=ACT.KWinnerTakeAll(50,axis=1,activation_function=ACT.Sigmoid()),
                         initial_weights='AUTO',
                         initial_bias=0.0,
                         initial_offset=numx.mean(train_data,axis = 0).reshape(1,train_data.shape[1]),
                         connections=None,
                         dtype=numx.float64)
l2 = LAYER.FullConnLayer(input_dim = 100,
                         output_dim = 10,
                         activation_function=ACT.SoftMax(),
                         initial_weights='AUTO',
                         initial_bias=0.0,
                         initial_offset=0.0,
                         connections=None,
                         dtype=numx.float64)
# Call constructor of superclass
model = MODEL.Model([l1, l2])
trainer = TRAINER.GDTrainer(model)

# Train model
max_epochs =100
batch_size = 10
eps = 0.1
print 'Training'
for epoch in range(1, max_epochs + 1):
    train_data, train_label = npExt.shuffle_dataset(train_data, train_label)
    for b in range(0, train_data.shape[0], batch_size):
        #eps *= 0.99998
        trainer.train(data=train_data[b:b + batch_size, :],
                      labels=[None,train_label[b:b + batch_size, :]],
                      costs = [None,COST.CrossEntropyError],
                      reg_costs = [0.0,1.0],
                      momentum=[0.0]*model.num_layers,
                      epsilon = [eps]*model.num_layers,
                      update_offsets = [0.01]*model.num_layers,
                      corruptor = [CORR.Dropout(0.2),CORR.Dropout(0.5),None],
                      reg_L1Norm = [0.0]*model.num_layers,
                      reg_L2Norm = [0.0]*model.num_layers,
                      reg_sparseness  = [0.0]*model.num_layers,
                      desired_sparseness = [0.0]*model.num_layers,
                      costs_sparseness = [None]*model.num_layers,
                      restrict_gradient = [0.0]*model.num_layers,
                      restriction_norm = 'Mat')
    print epoch,'\t',eps,'\t',
    print numx.mean(npExt.compare_index_of_max(model.forward_propagate(train_data),train_label)),'\t',
    print numx.mean(npExt.compare_index_of_max(model.forward_propagate(test_data), test_label))
