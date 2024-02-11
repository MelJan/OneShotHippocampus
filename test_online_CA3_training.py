# Import MKL for multi threading
try:
    import mkl
except:
    print("MKL not found, please pip install for best performance.")
import numpy as numx
from os.path import isfile

import sys, os
# Needed to load existing files, otherwise modules structure is not found
sys.path.append(os.path.split(os.getcwd())[0])

from hippocampus.dataEvaluator import *
from hippocampus.modelEvaluator import *
from hippocampus.dataProvider import *
from hippocampus import hippoModel

import pydeep.ae.model as AEModel
import pydeep.ae.trainer as AETrainer
import pydeep.base.activationfunction as ACT
import pydeep.base.costfunction as COST

from pydeep.preprocessing import STANDARIZER, rescale_data
import pydeep.misc.io as IO
import pydeep.misc.visualization as VIS
from pydeep.base.corruptor import RandomPermutation
import matplotlib
matplotlib.use('Qt4Agg')

# Used when the EC data is generated artificially
class DummyAe(object):

    @staticmethod
    def encode(x):
        return x

    @staticmethod
    def decode(x):
        return x

# Set random seed
np.random.seed(42)

# Set number of thrads to use
mkl.set_num_threads(2)


# Loads CA3, DG, AE if existing
load = False

# Chose a dataset of UNCORR, CORR, CROSSCORR, MNIST, CIFAR
dataset = "UNCORR"#"UNCORR" #

model_label = 'figs/model_A'

# Trainign mode
mode = "online"  # "batch"#"online+stab"

# Constant learning rate or scaled by activation?
lr_const = False

# Factor for the model size
f = 10

# Activity levels
CA3_activity = 0.2

# Set the capacity of the network
s1 = 10
s2 = 10 * f
CA3_capacity = s1 * s2

print "batch: " + str(mode)
print "lr: " + str(lr_const)


#############################################################################################
### Additional changes only if you know what zou change                                   ###
#############################################################################################

# Set the layer sizes depending on f
v1 = d1 = 11
v2 = d2 = 10 * f
EC_dim = v1 * v2
c1 = 25
c2 = 10 * f
CA3_dim = c1 * c2
DG_dim = 1200 * f
CA1_dim = None


# Number of epochs
epochs = 1

#############################################################################################
### No changes from here on                                                                ###
#############################################################################################

step_list = [1]#,5,25,100,250,CA3_capacity/2]
epsilons = [0.01]

if dataset == "UNCORR":
    intrinsic_sequence = generate_binary_random_sequence(CA3_capacity, CA3_dim, numx.int32(CA3_dim * CA3_activity))
elif dataset == 'CROSSCORR':
    intrinsic_sequence = generate_correlated_binary_random_sequence(CA3_capacity, CA3_dim,
                                                                    numx.int32(CA3_dim * CA3_activity),
                                                                    numx.int32(CA3_dim / 10.0))
result_base = calculate_correlation_two_sequences(intrinsic_sequence,
                         numx.tile(numx.mean(intrinsic_sequence, axis=0).reshape(intrinsic_sequence.shape[1], 1),
                                   intrinsic_sequence.shape[0]).T).reshape(intrinsic_sequence.shape[0])

'''
for e,e_idx in zip(epsilons,range(len(epsilons))):


    np.random.seed(int(e * 42))
    CA3_CA3 = Layer.PredictiveAssociator(intrinsic_sequence)
    CA3_CA3.train(e, 0, epochs, 1)
    for s,s_idx in zip(step_list,range(len(step_list))):

        # Set random seed

        rec_CA3_states= np.roll(intrinsic_sequence, s, 0)
        for i in range(s):
            rec_CA3_states = CA3_CA3.calculate_output(rec_CA3_states)
        result= calculate_correlation_two_sequences(rec_CA3_states, intrinsic_sequence)


        np.save("result_"+str(e)+"_"+str(s),result)
'''
step_list = [1,2,5,25,100,200,500]#,5,25,100,250,CA3_capacity/2]
epsilons = [0.075,0.050,0.025,0.01]

for e,e_idx in zip(epsilons,range(len(epsilons))):

    legends = []
    VIS.figure(str(e))

    x = np.arange(1,CA3_capacity+1)
    for s,s_idx in zip(step_list,range(len(step_list))):
        result = np.load("result_" + str(e) + "_" + str(s) + ".npy")
        VIS.plot(x, result)

        if s_idx == 0:
            legends.append(str(s) + " transition ")
        else:
            legends.append(str(s) + " transitions")
    #VIS.plot(x, result_base, linestyle='-.',color="green")

    VIS.xlabel('Pattern index t (' + str(intrinsic_sequence.shape[0]) + ' = latest pattern)', fontsize=14)
    VIS.ylabel(r'$Corr(\mathbf{\dot{x}}_{t}^{CA3},\mathbf{x}_t^{CA3})$', fontsize=14)


    VIS.xlim(0, CA3_capacity)
    VIS.ylim(-0.01, 1.01)
    #VIS.xlim(0,intrinsic_sequence.shape[0]+1)
    #legends.append("Baseline")
    #VIS.legend(legends)


VIS.show()
