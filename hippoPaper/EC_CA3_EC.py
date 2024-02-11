# Import MKL for multi threading
try:
    import mkl
    # Set number of thrads to use
    mkl.set_num_threads(4)
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
#mkl.set_num_threads(2)

# show data examples
show_data_examples = False

# show data examples
show_retrieved_patterns = True

# Loads CA3, DG, AE if existing
load = True

# Chose a dataset of UNCORR (Figure 11 b,d,f), CORR, CROSSCORR (Figure 10 b,d,e, Figure 11 a,c,e), MNIST, CIFAR
dataset = "CROSSCORR" #

model_label = 'figs/model_A'

# Trainign mode
mode = "online"  # "batch"#"online+stab"

# Constant learning rate or scaled by activation?
lr_const = False

# Use self enhancement at the vey end
self_enhance = False
if self_enhance:
    model_label = 'figs/model_A_dreaming'

# Factor for the model size
f = 10

# Activity levels
EC_activity = 0.35
CA3_activity = 0.2
DG_activity = None
CA1_activity = None

# Set the capacity of the network
s1 = 10
s2 = 10 * f
CA3_capacity = s1 * s2

print("batch: " + str(mode))
print("lr: " + str(lr_const))
print("self enhance: " + str(self_enhance))

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

# Number of trials to perform
trials = 1

# Number of stabalization steps in case of online+stab
stab_step = 10

# Number of epochs
epochs = 1

# Amount of noise to be used in test and for AE training
corr = RandomPermutation(0.1)

# Relaxes the dynamics on the pattern by interating x times back and forth
CA3_completion_loops = 0

# Add npise to the input during pattern storage
disturbance = 0.0

# Number of updates on the current datapoint
num_updates = 1

# Binaroze output
binarize_output = False

# No Weight deacay
L1_decay = 0.0000
L2_decay = 0.0000

# Ste the learnign rate
if lr_const:
    epsilon_enc = 0.1
    epsilon_dec = 0.1
else:
    epsilon_enc = 0.2/f
    epsilon_dec = 0.2/f

print(epochs, epsilon_enc, epsilon_dec)

#############################################################################################
### No changes from here on                                                                ###
#############################################################################################

models = []
datasets = []
datasets_next = []
intrinsics = []
intrinsics_next = []
datasets_noisy = []
intrinsics_noisy = []

for t in range(trials):

    # Set random seed
    np.random.seed(t + 42)

    # Generate or load input sequence
    if dataset == 'UNCORR':
        data = generate_binary_random_sequence(2 * CA3_capacity, EC_dim, EC_dim * EC_activity)
        train_data = data[0:CA3_capacity]
        test_data = data[CA3_capacity:2 * CA3_capacity]
        ae = DummyAe()
        print("Uncorr")
    elif dataset == 'CORR':
        data = generate_correlated_binary_random_patterns(2 * CA3_capacity, EC_dim, numx.int32(EC_dim * EC_activity),
                                                          numx.int32(EC_dim / 10.0))
        train_data = data[0:CA3_capacity]
        test_data = data[CA3_capacity:2 * CA3_capacity]
        ae = DummyAe()
        print("Corr")
    elif dataset == 'CROSSCORR':
        data = generate_correlated_binary_random_sequence(2 * CA3_capacity, EC_dim, numx.int32(EC_dim * EC_activity),
                                                          numx.int32(EC_dim / 10.0))
        train_data = data[0:CA3_capacity]
        test_data = data[CA3_capacity:2 * CA3_capacity]
        ae = DummyAe()
        print("Pairwise corr")
    elif dataset == 'MNIST':
        data = io.load_mnist("mnist.pkl.gz", False)
        data = numx.vstack((data[0], data[2]))
        print("Mnist")
        if isfile("AE_MNIST_" + str(EC_dim)) and load:
            ae = io.load_object("AE_MNIST_" + str(EC_dim))
        else:
            print("Train auto encoder on MNIST data.")
            # Create trainer and experiments
            ae = AEModel.AutoEncoder(number_visibles=784,
                                     number_hiddens=EC_dim,
                                     data=data,
                                     visible_activation_function=ACT.Sigmoid(),
                                     hidden_activation_function=ACT.Step(),
                                     cost_function=COST.CrossEntropyError(),
                                     initial_weights='AUTO',
                                     initial_visible_bias='AUTO',
                                     initial_hidden_bias='AUTO',
                                     initial_visible_offsets='AUTO',
                                     initial_hidden_offsets='AUTO',
                                     dtype=numx.float64)
            trainer = AETrainer.GDTrainer(ae)
            eps = 0.1
            bs = 10
            mom = 0.0
            ae_data = data
            for e in range(0, 10):
                ae_data = numx.random.permutation(ae_data)
                for b in range(0, ae_data.shape[0], bs):
                    rec = ae.encode(ae_data[b:b + bs, :])
                    trainer.train(data=ae_data[b:b + bs, :],
                                  num_epochs=1,
                                  epsilon=eps,
                                  momentum=mom,
                                  update_visible_offsets=0.0,
                                  update_hidden_offsets=0.01,
                                  reg_sparseness=0.0,
                                  desired_sparseness=EC_activity)
                    ae.bh -= eps * 1 * np.mean(rec - EC_activity, axis=0).reshape(1, EC_dim)
                enc_data = ae.encode(ae_data)
                print(e,numx.mean((numx.abs(ae.decode(enc_data) - ae_data))), 
                    numx.mean(numx.abs(enc_data)))
            io.save_object(ae, "AE_MNIST_" + str(EC_dim))
        train_data = data[0:CA3_capacity]
        test_data = data[CA3_capacity:2 * CA3_capacity]
        d1 = d2 = 28
    elif dataset == 'CIFAR':
        print("Cifar")
        data = io.load_cifar("cifar-10-python.tar.gz", True)[0]
        data = data  # [0:2*CA3_capacity]
        data = rescale_data(data, 0, 1)
        train_data = data[0:CA3_capacity]
        test_data = data[CA3_capacity:2 * CA3_capacity]
        # data = train_data
        if isfile("AE_CIFAR_" + str(EC_dim)) and load:
            ae = io.load_object("AE_CIFAR_" + str(EC_dim))
        else:
            # Create trainer and experiments
            print("Train auto encoder on CIFAR data.")
            ae = AEModel.AutoEncoder(number_visibles=1024,
                                     number_hiddens=EC_dim,
                                     data=data,
                                     visible_activation_function=ACT.Sigmoid(),
                                     hidden_activation_function=ACT.Step(),
                                     cost_function=COST.CrossEntropyError(),
                                     initial_weights='AUTO',
                                     initial_visible_bias='AUTO',
                                     initial_hidden_bias='AUTO',
                                     initial_visible_offsets='AUTO',
                                     initial_hidden_offsets='AUTO',
                                     dtype=numx.float64)
            trainer = AETrainer.GDTrainer(ae)
            eps = 0.01
            bs = 100
            mom = 0.9
            ae_data = data
            for e in range(0, 100):
                ae_data = numx.random.permutation(ae_data)
                for b in range(0, ae_data.shape[0], bs):
                    rec = ae.encode(ae_data[b:b + bs, :])
                    trainer.train(data=ae_data[b:b + bs, :],
                                  num_epochs=1,
                                  epsilon=eps,
                                  momentum=mom,
                                  update_visible_offsets=0.0,
                                  update_hidden_offsets=0.01,
                                  reg_sparseness=0.0,
                                  desired_sparseness=EC_activity)
                    ae.bh -= eps * 1 * np.mean(rec - EC_activity, axis=0).reshape(1, EC_dim)
                enc_data = ae.encode(ae_data)
                print(e,numx.mean((numx.abs(ae.decode(enc_data) - ae_data))), numx.mean(numx.abs(enc_data)))
            io.save_object(ae, "AE_CIFAR_" + str(EC_dim))
        d1 = d2 = 32
    else:
        print("Choose correct dataset")

    # Get EC data
    EC_data_train = ae.encode(train_data)
    EC_data_test = ae.encode(test_data)


    print("Correlation train", caluclate_average_correlation(EC_data_train))
    print("Correlation test", caluclate_average_correlation(EC_data_test))
    print("Pairwise correlation train", caluclate_average_temp_correlation(EC_data_train))
    print("Pairwise correlation test", caluclate_average_temp_correlation(EC_data_test))
    print("EC_dim\tCA3_dim\tDG_dim\tEC_activity\tCA3_activity\tDG_activity\tCA3_capacity")
    print(EC_dim, "\t",CA3_dim, "\t",DG_dim, "\t",EC_activity, "\t",CA3_activity, "\t",DG_activity, "\t",CA3_capacity)

    # Specify Hippocampus model
    hippo = hippoModel.HippoECCA3EC(EC_dim=EC_dim,
                                      EC_activity=EC_activity,
                                      CA3_dim=CA3_dim,
                                      CA3_activity=CA3_activity,
                                      CA3_capacity=CA3_capacity,
                                      CA3_completion_loops=CA3_completion_loops,
                                      binarize_output=binarize_output,
                                      load=load)

    # Store all datapoint one after the other in the model, either online(default), batch (just proof of concept),
    # or online + stab (experimental)
    if mode == "online":
        for e in range(epochs):
            for i in range(EC_data_train.shape[0]):
                print(i)
                hippo.store_data_point(EC_data_train[i],
                                       epsilon_enc=epsilon_enc,
                                       epsilon_dec=epsilon_dec,
                                       disturbance=disturbance,
                                       iterations=num_updates,
                                       l1norm=L1_decay,
                                       l2norm=L2_decay)
    elif mode == "batch":
        for e in range(epochs):
            hippo.store_data_point(EC_data_train,
                                   epsilon_enc=epsilon_enc,
                                   epsilon_dec=epsilon_dec,
                                   disturbance=disturbance,
                                   iterations=num_updates,
                                   l1norm=L1_decay,
                                   l2norm=L2_decay)
    elif mode == "online+stab":
        for e in range(epochs):
            for i in range(EC_data_train.shape[0]):
                hippo.store_data_point(EC_data_train[i],
                                       epsilon_enc=epsilon_enc,
                                       epsilon_dec=epsilon_dec,
                                       disturbance=disturbance,
                                       iterations=num_updates,
                                       l1norm=L1_decay,
                                       l2norm=L2_decay)
                if i % stab_step == 0:
                    hippo.stabalize_memory(epochs=100, epsilon=0.1)
    else:
        print("CHOOSE VALID TRAIN SETUP")

    # If true EC->CA or DG->CA3 will be retrained on intrinsic reconstructions
    if self_enhance:
        print("enhancing")
        hippo.stabalize_memory(epochs=10,
                               epsilon=0.1)

    # Store data, model and resutls or the current trial
    models.append(hippo)
    datasets.append(EC_data_train)
    datasets_next.append(numx.roll(EC_data_train, -1, 0))
    intrinsics.append(hippo.CA3_CA3._training_sequence)
    intrinsics_next.append(numx.roll(hippo.CA3_CA3._training_sequence, -1, 0))
    datasets_noisy.append(corr.corrupt(EC_data_train))
    intrinsics_noisy.append(corr.corrupt(hippo.CA3_CA3._training_sequence))

if show_data_examples:
    # Visualize results for a single datapoint
    point = raw = train_data[CA3_capacity-2:CA3_capacity-1,:]
    print(numx.mean(raw))
    raw = point
    print(numx.mean(raw))
    org_data = VIS.tile_matrix_rows(matrix=raw.T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=1,
                                    num_tiles_y=1,
                                    border_size=0,
                                    normalized=False)
    print(numx.mean(raw))
    raw =train_data
    print(numx.mean(raw))

    VIS.imshow_matrix(matrix=org_data,
                      windowtitle='input data')
    IO.save_image(org_data,'input_data','png')
    print(numx.mean(raw))
    raw =ae.encode(point)
    print(numx.mean(raw))
    org_data = VIS.tile_matrix_rows(matrix=raw.T,
                                    tile_width=22,
                                    tile_height=5*f,
                                    num_tiles_x=1,
                                    num_tiles_y=1,
                                    border_size=0,
                                    normalized=False)
    VIS.imshow_matrix(matrix=org_data,
                      windowtitle='EC data')
    IO.save_image(org_data,'ec_data','png')
    print(numx.mean(raw))
    raw =ae.encode(train_data)
    print(numx.mean(raw))

    raw = hippo.EC_CA3.calculate_output(ae.encode(point))
    org_data = VIS.tile_matrix_rows(matrix=raw.T,
                                    tile_width=25,
                                    tile_height=10*f,
                                    num_tiles_x=1,
                                    num_tiles_y=1,
                                    border_size=0,
                                    normalized=False)
    VIS.imshow_matrix(matrix=org_data,
                      windowtitle='CA3 data')
    IO.save_image(org_data,'ca3_data','png')
    print(numx.mean(raw))
    raw = hippo.EC_CA3.calculate_output(ae.encode(train_data))
    print(numx.mean(raw))

    raw = hippo.CA3_CA3.calculate_output(hippo.EC_CA3.calculate_output(ae.encode(point)))
    org_data = VIS.tile_matrix_rows(matrix=raw.T,
                                    tile_width=25,
                                    tile_height=10*f,
                                    num_tiles_x=1,
                                    num_tiles_y=1,
                                    border_size=0,
                                    normalized=False)
    VIS.imshow_matrix(matrix=org_data,
                      windowtitle='CA3 next')
    IO.save_image(org_data,'ca3_next','png')
    print(numx.mean(raw))
    raw = hippo.CA3_CA3.calculate_output(hippo.EC_CA3.calculate_output(ae.encode(train_data)))
    print(numx.mean(raw))

    raw = hippo.CA3_EC.calculate_output(hippo.CA3_CA3.calculate_output(hippo.EC_CA3.calculate_output(ae.encode(point))))
    org_data = VIS.tile_matrix_rows(matrix=raw.T,
                                    tile_width=22,
                                    tile_height=5*f,
                                    num_tiles_x=1,
                                    num_tiles_y=1,
                                    border_size=0,
                                    normalized=False)
    VIS.imshow_matrix(matrix=org_data,
                      windowtitle='rec ec')
    IO.save_image(org_data,'rec_ec','png')
    print(numx.mean(raw))
    raw = ae.decode(hippo.CA3_EC.calculate_output(hippo.CA3_CA3.calculate_output(hippo.EC_CA3.calculate_output(ae.encode(train_data)))))
    print(numx.mean(raw))

    raw = ae.decode(hippo.CA3_EC.calculate_output(hippo.CA3_CA3.calculate_output(hippo.EC_CA3.calculate_output(ae.encode(point)))))
    org_data = VIS.tile_matrix_rows(matrix=raw.T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=1,
                                    num_tiles_y=1,
                                    border_size=0,
                                    normalized=False)
    VIS.imshow_matrix(matrix=org_data,
                      windowtitle='rec data')
    IO.save_image(numx.float64(org_data),'rec_data','png')
    print(numx.mean(raw))
    raw = ae.decode(hippo.CA3_EC.calculate_output(hippo.CA3_CA3.calculate_output(hippo.EC_CA3.calculate_output(ae.encode(train_data)))))
    print(numx.mean(raw))

    VIS.show()


# Show performance curves of the subregions
get_performance_encoder(models=models, EC_states=datasets, CA3_targets=intrinsics,
                        measure=calculate_correlation_two_sequences)

vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'encode.eps', format='eps')

get_performance_decoder(models=models, CA3_states=intrinsics, EC_targets=datasets,
                        measure=calculate_correlation_two_sequences)

vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'decode.eps', format='eps')

get_performance_encoder_decoder(models=models, EC_states=datasets, EC_targets=datasets,
                                measure=calculate_correlation_two_sequences)

vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'encode_decode.eps', format='eps')

get_performance_dynamic(models=models, CA3_states=intrinsics, CA3_targets=intrinsics_next,
                        measure=calculate_correlation_two_sequences)

#get_performance_dynamic(models=models, CA3_states=intrinsics_noisy, CA3_targets=intrinsics_next,
#                        measure=calculate_correlation_two_sequences)
get_performance_encoder_dynamic(models=models, EC_states=datasets, CA3_targets=intrinsics_next,
                                measure=calculate_correlation_two_sequences)
vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'encoder_dynamic_one_step.eps', format='eps')

get_performance_encoder_dynamic_5(models=models, EC_states=datasets, CA3_targets=numx.roll(intrinsics_next, -4, 1),
                                measure=calculate_correlation_two_sequences)
vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'encoder_dynamic_five_step.eps', format='eps')


get_performance_dynamic_decoder(models=models, CA3_states=intrinsics, EC_targets=datasets_next,
                                measure=calculate_correlation_two_sequences)

get_performance_encoder_dynamic_decoder(models=models, EC_states=datasets, EC_targets=datasets_next,
                                        measure=calculate_correlation_two_sequences)


steps = 5
get_performance_intrinsic(models=models, EC_states=datasets, EC_targets=numx.roll(datasets, -steps, 1), steps=steps,
                          measure=calculate_correlation_two_sequences)
vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'intrinsic_'+str(steps)+'_steps.eps', format='eps')

steps = 20
get_performance_intrinsic(models=models, EC_states=datasets, EC_targets=numx.roll(datasets, -steps, 1), steps=steps,
                          measure=calculate_correlation_two_sequences)
vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'intrinsic_'+str(steps)+'_steps.eps', format='eps')

steps = CA3_capacity
get_performance_intrinsic(models=models, EC_states=datasets, EC_targets=numx.roll(datasets, -steps, 1), steps=steps,
                          measure=calculate_correlation_two_sequences)
vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'intrinsic_'+str(steps)+'_steps.eps', format='eps')


steps = 1
get_performance_full_loop(models=models, EC_states=datasets, EC_targets=numx.roll(datasets, -steps, 1), steps=steps,
                          measure=calculate_correlation_two_sequences)
vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'full_loop_'+str(steps)+'_steps.eps', format='eps')

steps = 2
get_performance_full_loop(models=models, EC_states=datasets, EC_targets=numx.roll(datasets, -steps, 1), steps=steps,
                          measure=calculate_correlation_two_sequences)
vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'full_loop_'+str(steps)+'_steps.eps', format='eps')

steps = 5
get_performance_full_loop(models=models, EC_states=datasets, EC_targets=numx.roll(datasets, -steps, 1), steps=steps,
                          measure=calculate_correlation_two_sequences)
vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'full_loop_'+str(steps)+'_steps.eps', format='eps')

steps = 100
get_performance_full_loop(models=models, EC_states=datasets, EC_targets=numx.roll(datasets, -steps, 1), steps=steps,
                          measure=calculate_correlation_two_sequences)
vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'full_loop_'+str(steps)+'_steps.eps', format='eps')

steps = CA3_capacity
get_performance_full_loop(models=models, EC_states=datasets, EC_targets=numx.roll(datasets, -steps, 1), steps=steps,
                          measure=calculate_correlation_two_sequences)
vis.savefig(model_label+'_'+str.lower(dataset)+'_'+'full_loop_'+str(steps)+'_steps.eps', format='eps')


vis.show()

if show_retrieved_patterns:
    # Show reconstruction for cues
    org_data = VIS.tile_matrix_rows(matrix=train_data.T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=s1,
                                    num_tiles_y=s2,
                                    border_size=1,
                                    normalized=True)
    VIS.imshow_matrix(matrix=org_data,
                      windowtitle='VC data')

    rec_data = VIS.tile_matrix_rows(matrix=ae.decode(EC_data_train).T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=s1,
                                    num_tiles_y=s2,
                                    border_size=1,
                                    normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='EC reconstructed data')
    '''
    rec_data = VIS.tile_matrix_rows(matrix=ae.decode(models[0].encode_next_decode(numx.roll(EC_data_train, 1, 0))).T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=s1,
                                    num_tiles_y=s2,
                                    border_size=1,
                                    normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='next')
    '''
    rec_data = VIS.tile_matrix_rows(matrix=ae.decode(models[0].encode_decode(EC_data_train)).T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=s1,
                                    num_tiles_y=s2,
                                    border_size=1,
                                    normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Reconstruction without intrinsic dynamics')

    idx = 10
    res = models[0].full_intrinsic_loop_encode_next_decode(EC_data_train[idx])
    res = numx.roll(res, idx + 1, 0)
    res = ae.decode(res)
    res[idx,:] = 1-train_data[idx]
    rec_data = VIS.tile_matrix_rows(
        matrix=res.T,
        tile_width=d1,
        tile_height=d2,
        num_tiles_x=s1,
        num_tiles_y=s2,
        border_size=1,
        normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Full intrinsic reconstruction for cue ' + str(idx))

    idx = 10
    #sample = numx.abs(train_data[idx].reshape(1,d2*d1)-numx.random.binomial(n=1,p=0.1,size=784).reshape(1,784))
    #print(sample.shape)
    #print(numx.sum(sample),numx.sum(numx.abs(sample)))
    #print(numx.sum(train_data[idx]),numx.sum(numx.abs(train_data[idx])))
    sample = corr.corrupt(train_data[idx].reshape(1,d2*d1))
    res = ae.decode(numx.roll(models[0].full_intrinsic_loop_encode_next_decode(ae.encode(sample)), idx + 1, 0))
    res[idx,:]= 1-sample
    rec_data = VIS.tile_matrix_rows(
        matrix=res.T,
        tile_width=d1,
        tile_height=d2,
        num_tiles_x=s1,
        num_tiles_y=s2,
        border_size=1,
        normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Full intrinsic reconstruction for cue ' + str(idx)+ ' 10% input noise ')


    idx = 10
    sample = corr.corrupt(ae.encode(train_data[idx].reshape(1,d2*d1)))
    #sample = numx.abs(ae.encode(train_data[idx].reshape(1,d2*d1))-numx.random.binomial(n=1,p=0.1,size=EC_dim).reshape(1,EC_dim))
    #print(sample.shape)
    #print(numx.sum(sample),numx.sum(numx.abs(sample)))
    #print(numx.sum(ae.encode(train_data[idx].reshape(1,d2*d1))),numx.sum(numx.abs(ae.encode(train_data[idx].reshape(1,d2*d1)))))
    res = numx.roll(models[0].full_intrinsic_loop_encode_next_decode(sample), idx + 1, 0)
    res[idx,:]= sample
    res = ae.decode(res)
    res[idx,:]=1-res[idx,:]
    rec_data = VIS.tile_matrix_rows(
        matrix=res.T,
        tile_width=d1,
        tile_height=d2,
        num_tiles_x=s1,
        num_tiles_y=s2,
        border_size=1,
        normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Full intrinsic reconstruction for cue ' + str(idx)+ ' 210% EC noise ')

    idx = 190
    res = ae.decode(numx.roll(models[0].full_intrinsic_loop_encode_next_decode(EC_data_train[idx]), idx + 1, 0))
    res[idx,:]= 1-train_data[idx]
    rec_data = VIS.tile_matrix_rows(
        matrix=res.T,
        tile_width=d1,
        tile_height=d2,
        num_tiles_x=s1,
        num_tiles_y=s2,
        border_size=1,
        normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Full intrinsic reconstruction for cue ' + str(idx))

    idx = 14
    res = ae.decode(numx.roll(models[0].full_intrinsic_loop_encode_next_decode(EC_data_train[idx]), idx + 1, 0))
    res[idx,:]= 1-train_data[idx]
    rec_data = VIS.tile_matrix_rows(
        matrix=res.T,
        tile_width=d1,
        tile_height=d2,
        num_tiles_x=s1,
        num_tiles_y=s2,
        border_size=1,
        normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Full intrinsic reconstruction for cue ' + str(idx))

    idx = 121
    res = ae.decode(numx.roll(models[0].full_intrinsic_loop_encode_next_decode(EC_data_train[idx]), idx + 1, 0))
    res[idx,:]= 1-train_data[idx]
    rec_data = VIS.tile_matrix_rows(
        matrix=res.T,
        tile_width=d1,
        tile_height=d2,
        num_tiles_x=s1,
        num_tiles_y=s2,
        border_size=1,
        normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Full intrinsic reconstruction for cue ' + str(idx))

    res = ae.decode(numx.roll(models[0].full_intrinsic_loop_encode_next_decode(EC_data_train[idx]), idx - 38, 0))
    res[idx-39,:]= 1-train_data[idx]
    rec_data = VIS.tile_matrix_rows(
        matrix=res.T,
        tile_width=d1,
        tile_height=d2,
        num_tiles_x=s1,
        num_tiles_y=s2,
        border_size=1,
        normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Full intrinsic reconstruction for cue ' + str(idx) + " SHIFTED by 39")

    idx = 0
    res = ae.decode(numx.roll(models[0].full_intrinsic_loop_encode_next_decode(EC_data_test[idx]), idx + 1, 0))
    res[idx+1,:]= 1-test_data[idx]
    rec_data = VIS.tile_matrix_rows(
        matrix=res.T,
        tile_width=d1,
        tile_height=d2,
        num_tiles_x=s1,
        num_tiles_y=s2,
        border_size=1,
        normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Full intrinsic reconstruction for test cue ')

    idx = 0
    res = ae.decode(numx.roll(models[0].full_intrinsic_loop_encode_next_decode(EC_data_test[idx]), idx, 0))
    res[idx+1,:]= 1-test_data[idx]
    rec_data = VIS.tile_matrix_rows(
        matrix=res.T,
        tile_width=d1,
        tile_height=d2,
        num_tiles_x=s1,
        num_tiles_y=s2,
        border_size=1,
        normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Full intrinsic reconstruction for test cue '+str(idx))

    idx = 5
    res = ae.decode(numx.roll(models[0].full_intrinsic_loop_encode_next_decode(EC_data_test[idx]), idx, 0))
    res[idx+1,:]= 1-test_data[idx]
    rec_data = VIS.tile_matrix_rows(
        matrix=res.T,
        tile_width=d1,
        tile_height=d2,
        num_tiles_x=s1,
        num_tiles_y=s2,
        border_size=1,
        normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Full intrinsic reconstruction for test cue '+str(idx))

    idx = 10
    res = ae.decode(numx.roll(models[0].full_intrinsic_loop_encode_next_decode(EC_data_test[idx]), idx, 0))
    res[idx+1,:]= 1-test_data[idx]
    rec_data = VIS.tile_matrix_rows(
        matrix=res.T,
        tile_width=d1,
        tile_height=d2,
        num_tiles_x=s1,
        num_tiles_y=s2,
        border_size=1,
        normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Full intrinsic reconstruction for test cue '+str(idx))

    idx = 20
    res = ae.decode(numx.roll(models[0].full_intrinsic_loop_encode_next_decode(EC_data_test[idx]), idx, 0))
    res[idx+1,:]= 1-test_data[idx]
    rec_data = VIS.tile_matrix_rows(
        matrix=res.T,
        tile_width=d1,
        tile_height=d2,
        num_tiles_x=s1,
        num_tiles_y=s2,
        border_size=1,
        normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Full intrinsic reconstruction for test cue '+str(idx))

    '''
    print(models[0].full_intrinsic_loop_encode_next_decode(EC_data_train).shape)
    rec_data = VIS.tile_matrix_rows(matrix=ae.decode(
        models[0].full_intrinsic_loop_encode_next_decode(EC_data_train)[(f * f * 10000 - f * 100):(f * f * 10000), :]).T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=s1,
                                    num_tiles_y=s2,
                                    border_size=1,
                                    normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='full intrinsic all')
    '''
VIS.show()
