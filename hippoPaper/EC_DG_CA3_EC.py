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
from pydeep.base.corruptor import RandomPermutation, BinaryNoise

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



# show data examples
show_data_examples = False

# show data examples
show_retrieved_patterns = True

# Loads CA3, DG, AE if existing
load = True

# Chose a dataset of UNCORR CORR, CROSSCORR (Figure 10 a,c), MNIST, CIFAR
dataset = "MNIST"

model_label = 'figs/model_B'

# Trainign mode
mode = "online"  # "batch"#"online+stab"

# Constant learning rate or scaled by activation?
lr_const = False

# Use self enhancement at the vey end
self_enhance = False
if self_enhance:
    model_label = 'figs/model_B_dreaming'

# Factor for the model size
f = 10

# Activity levels
EC_activity = 0.35
CA3_activity = 0.2 #(Figure 10,11 0.2, Figure a,c,e 0.1, Figure 14 b,d,f 0.032)
DG_activity = 0.03
CA1_activity = None

# Set the capacity of the network
s1 = 10
s2 = 10 * f
CA3_capacity = s1 * s2

print "batch: " + str(mode)
print "lr: " + str(lr_const)
print "self enhance: " + str(self_enhance)

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

print epochs, epsilon_enc, epsilon_dec

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
        print "Uncorr"
    elif dataset == 'CORR':
        data = generate_correlated_binary_random_patterns(2 * CA3_capacity, EC_dim, numx.int32(EC_dim * EC_activity),
                                                          numx.int32(EC_dim / 10.0))
        train_data = data[0:CA3_capacity]
        test_data = data[CA3_capacity:2 * CA3_capacity]
        ae = DummyAe()
        print "Corr"
    elif dataset == 'CROSSCORR':
        data = generate_correlated_binary_random_sequence(2 * CA3_capacity, EC_dim, numx.int32(EC_dim * EC_activity),
                                                          numx.int32(EC_dim / 10.0))
        train_data = data[0:CA3_capacity]
        test_data = data[CA3_capacity:2 * CA3_capacity]
        ae = DummyAe()
        print "Pairwise corr"
    elif dataset == 'MNIST':
        data = io.load_mnist("mnist.pkl.gz", False)
        data = numx.vstack((data[0], data[2]))
        print "Mnist"
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
            eps = 0.01
            bs = 100
            mom = 0.9
            ae_data = data
            for e in range(0, 10):
                print e,
                ae_data = numx.random.permutation(ae_data)
                for b in range(0, ae_data.shape[0], bs):
                    rec = ae.encode(ae_data[b:b + bs, :])
                    trainer.train(data=ae_data[b:b + bs, :],
                                  num_epochs=1,
                                  epsilon=eps,
                                  momentum=mom,
                                  update_visible_offsets=0.0,
                                  update_hidden_offsets=0.01)
                    ae.bh -= eps * 1 * np.mean(rec - EC_activity, axis=0).reshape(1, EC_dim)
                enc_data = ae.encode(ae_data)
                print numx.mean((numx.abs(ae.decode(enc_data) - ae_data))), \
                    numx.mean(numx.abs(enc_data))
            io.save_object(ae, "AE_MNIST_" + str(EC_dim))
        train_data = data[0:CA3_capacity]
        test_data = data[CA3_capacity:2 * CA3_capacity]
        d1 = d2 = 28
    elif dataset == 'CIFAR':
        print "Cifar"
        data = io.load_cifar("cifar-10-python.tar.gz", True)[0]
        data = data#[0:2*CA3_capacity]
        data = rescale_data(data,0,1)
        train_data = data[0:CA3_capacity]
        test_data = data[CA3_capacity:2 * CA3_capacity]
        #data = train_data
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
            for e in range(0, 10):
                print e,
                ae_data = numx.random.permutation(ae_data)
                for b in range(0, ae_data.shape[0], bs):
                    rec = ae.encode(ae_data[b:b + bs, :])
                    trainer.train(data=ae_data[b:b + bs, :],
                                  num_epochs=1,
                                  epsilon=eps,
                                  momentum=mom,
                                  update_visible_offsets=0.0,
                                  update_hidden_offsets=0.01)
                    ae.bh -= eps * 1 * np.mean(rec - EC_activity, axis=0).reshape(1, EC_dim)
                enc_data = ae.encode(ae_data)
                print numx.mean((numx.abs(ae.decode(enc_data) - ae_data))), \
                    numx.mean(numx.abs(enc_data))
            io.save_object(ae, "AE_CIFAR_" + str(EC_dim))
        d1 = d2 = 32
    else:
        print "Choose correct dataset"

    # Get EC data
    EC_data_train = ae.encode(train_data[0:CA3_capacity])
    EC_data_test = ae.encode(test_data[0:CA3_capacity])


    print "Correlation train", caluclate_average_correlation(EC_data_train)
    print "Correlation test", caluclate_average_correlation(EC_data_test)
    print "Pairwise correlation train", caluclate_average_temp_correlation(EC_data_train)
    print "Pairwise correlation test", caluclate_average_temp_correlation(EC_data_test)
    print "EC_dim\tCA3_dim\tDG_dim\tEC_activity\tCA3_activity\tDG_activity\tCA3_capacity"
    print EC_dim, "\t",CA3_dim, "\t",DG_dim, "\t",EC_activity, "\t",CA3_activity, "\t",DG_activity, "\t",CA3_capacity

    # Specify Hippocampus model
    hippo = hippoModel.HippoECDGCA3EC(EC_dim=EC_dim,
                                      EC_activity=EC_activity,
                                      DG_dim=DG_dim,
                                      DG_activity=DG_activity,
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
                print i
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
        print "CHOOSE VALID TRAIN SETUP"

    # If true EC->CA or DG->CA3 will be retrained on intrinsic reconstructions
    if self_enhance:
        print "enhancing"
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
    print numx.mean(raw)
    raw = point
    print numx.mean(raw)
    org_data = VIS.tile_matrix_rows(matrix=raw.T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=1,
                                    num_tiles_y=1,
                                    border_size=0,
                                    normalized=False)
    print numx.mean(raw)
    raw =train_data
    print numx.mean(raw)

    VIS.imshow_matrix(matrix=org_data,
                      windowtitle='input data')
    IO.save_image(org_data,'input_data','png')
    print numx.mean(raw)
    raw =ae.encode(point)
    print numx.mean(raw)
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
    print numx.mean(raw)
    raw =ae.encode(train_data)
    print numx.mean(raw)

    raw = hippo.EC_DG.calculate_output(ae.encode(point))
    org_data = VIS.tile_matrix_rows(matrix=raw.T,
                                    tile_width=50,
                                    tile_height=24*f,
                                    num_tiles_x=1,
                                    num_tiles_y=1,
                                    border_size=0,
                                    normalized=False)
    VIS.imshow_matrix(matrix=org_data,
                      windowtitle='DG data')
    IO.save_image(org_data,'dg_data','png')
    print numx.mean(raw)
    raw =hippo.EC_DG.calculate_output(ae.encode(train_data))
    print numx.mean(raw)

    raw = hippo.DG_CA3.calculate_output(hippo.EC_DG.calculate_output(ae.encode(point)))
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
    print numx.mean(raw)
    raw = hippo.DG_CA3.calculate_output(hippo.EC_DG.calculate_output(ae.encode(train_data)))
    print numx.mean(raw)

    raw = hippo.CA3_CA3.calculate_output(hippo.DG_CA3.calculate_output(hippo.EC_DG.calculate_output(ae.encode(point))))
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
    print numx.mean(raw)
    raw = hippo.DG_CA3.calculate_output(hippo.EC_DG.calculate_output(ae.encode(train_data)))
    print numx.mean(raw)

    raw = hippo.CA3_EC.calculate_output(hippo.CA3_CA3.calculate_output(hippo.DG_CA3.calculate_output(hippo.EC_DG.calculate_output(ae.encode(point)))))
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
    print numx.mean(raw)
    raw = ae.decode(hippo.CA3_EC.calculate_output(hippo.CA3_CA3.calculate_output(hippo.DG_CA3.calculate_output(hippo.EC_DG.calculate_output(ae.encode(train_data))))))
    print numx.mean(raw)

    raw = ae.decode(hippo.CA3_EC.calculate_output(hippo.CA3_CA3.calculate_output(hippo.DG_CA3.calculate_output(hippo.EC_DG.calculate_output(ae.encode(point))))))
    org_data = VIS.tile_matrix_rows(matrix=raw.T,
                                    tile_width=28,
                                    tile_height=28,
                                    num_tiles_x=1,
                                    num_tiles_y=1,
                                    border_size=0,
                                    normalized=False)
    VIS.imshow_matrix(matrix=org_data,
                      windowtitle='rec data')
    IO.save_image(numx.float64(org_data),'rec_data','png')
    print numx.mean(raw)
    raw = ae.decode(hippo.CA3_EC.calculate_output(hippo.CA3_CA3.calculate_output(hippo.DG_CA3.calculate_output(hippo.EC_DG.calculate_output(ae.encode(train_data))))))
    print numx.mean(raw)

    VIS.show()

exit()

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

'''
steps = 1
get_perfoIliB42mVhK*#rmance_full_loop(models=models, EC_states=datasets, EC_targets=numx.roll(datasets, -steps, 1), steps=steps,
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

'''
vis.show()

# Show reconstruction for cues
if show_retrieved_patterns:

    if f == 10:
        s1 = 40
        s2 = 25

    if f == 2:
        s1 = 25
        s2 = 8

    if f == 4:
        s1 = 25
        s2 = 16

    org_data = VIS.tile_matrix_rows(matrix=train_data.T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=s2,
                                    num_tiles_y=s1,
                                    border_size=1,
                                    normalized=True)
    VIS.imshow_matrix(matrix=org_data,
                      windowtitle='MSI data')
    if dataset=='MNIST':
        #IO.save_image(org_data, 'mnist_200_data', 'eps')
        VIS.imsave('mnist_200_data.pdf', org_data)
    elif dataset=='CIFAR':
        #IO.save_image(org_data, 'cifar_200_data', 'eps')
        VIS.imsave('cifar_200_data.pdf', org_data)

    ec_ground = ae.decode(ae.encode(train_data))
    rec_data = VIS.tile_matrix_rows(matrix=ec_ground.T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=s2,
                                    num_tiles_y=s1,
                                    border_size=1,
                                    normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='MSI -> EC -> MSI')
    if dataset=='MNIST':
        #IO.save_image(rec_data, 'mnist_200_ae_rec', 'eps')
        VIS.imsave('mnist_200_ae_rec.pdf', rec_data)
    elif dataset=='CIFAR':
        #IO.save_image(rec_data, 'cifar_200_ae_rec', 'eps')
        VIS.imsave('cifar_200_ae_rec.pdf', rec_data)


    ec_ground = ae.decode(models[0].decode(models[0].CA3_CA3._training_sequence))
    rec_data = VIS.tile_matrix_rows(matrix=ec_ground.T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=s2,
                                    num_tiles_y=s1,
                                    border_size=1,
                                    normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='CA3 -> EC -> MSI')
    if dataset=='MNIST':
        #IO.save_image(rec_data, 'mnist_200_ae_CA3_rec', 'eps')
        VIS.imsave('mnist_200_ae_CA3_rec.pdf',rec_data)
    elif dataset=='CIFAR':
        #IO.save_image(rec_data, 'cifar_200_ae_CA3_rec', 'eps')
        VIS.imsave('cifar_200_ae_CA3_rec.pdf',rec_data)


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

    reconstruction = ae.decode(models[0].encode_decode(EC_data_train))
    rec_data = VIS.tile_matrix_rows(matrix=reconstruction.T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=s2,
                                    num_tiles_y=s1,
                                    border_size=1,
                                    normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Reconstruction without intrinsic dynamics')
    '''
    intrinsic = ae.decode(models[0].full_intrinsic_loop_encode_next_decode_return_last(ae.encode(train_data)))
    rec_data = VIS.tile_matrix_rows(matrix=intrinsic.T,
                                    tile_width=d1,
                                    tile_height=d2,
                                    num_tiles_x=s2,
                                    num_tiles_y=s1,
                                    border_size=1,
                                    normalized=True)
    VIS.imshow_matrix(matrix=rec_data,
                      windowtitle='Reconstruction full intrinsic dynamics')



    if dataset=='MNIST':

        myres = numx.ones((rec_data.shape[0]+2,rec_data.shape[1]+2,3))
        myres[1:rec_data.shape[0]+1,1:rec_data.shape[1]+1,0]= rec_data
        myres[1:rec_data.shape[0]+1, 1:rec_data.shape[1]+1, 1] = rec_data
        myres[1:rec_data.shape[0]+1, 1:rec_data.shape[1]+1, 2] = rec_data
        val = 255

        myidx = 8
        xpos0 = (myidx / 25) * 29
        ypos0 = (myidx % 25) * 29
        xpos1 = xpos0+30
        ypos1 = ypos0+30
        myres[xpos0:xpos1, ypos0:ypos1, 2] = val
        myres[xpos0:xpos1, ypos0:ypos1, 0] = val

        myidx = 14
        xpos0 = (myidx / 25) * 29
        ypos0 = (myidx % 25) * 29
        xpos1 = xpos0+30
        ypos1 = ypos0+30
        myres[xpos0:xpos1, ypos0:ypos1, 2] = val
        myres[xpos0:xpos1, ypos0:ypos1, 0] = val

        myidx = 40
        xpos0 = (myidx / 25) * 29
        ypos0 = (myidx % 25) * 29
        xpos1 = xpos0+30
        ypos1 = ypos0+30
        myres[xpos0:xpos1, ypos0:ypos1, 2] = val
        myres[xpos0:xpos1, ypos0:ypos1, 0] = val

        myidx = 72
        xpos0 = (myidx / 25) * 29
        ypos0 = (myidx % 25) * 29
        xpos1 = xpos0+30
        ypos1 = ypos0+30
        myres[xpos0:xpos1, ypos0:ypos1, 2] = val
        myres[xpos0:xpos1, ypos0:ypos1, 0] = val

        myidx = 79
        xpos0 = (myidx / 25) * 29
        ypos0 = (myidx % 25) * 29
        xpos1 = xpos0+30
        ypos1 = ypos0+30
        myres[xpos0:xpos1, ypos0:ypos1, 2] = val
        myres[xpos0:xpos1, ypos0:ypos1, 0] = val

        myidx = 141
        xpos0 = (myidx / 25) * 29
        ypos0 = (myidx % 25) * 29
        xpos1 = xpos0+30
        ypos1 = ypos0+30
        myres[xpos0:xpos1, ypos0:ypos1, 1] = val
        myres[xpos0:xpos1, ypos0:ypos1, 2] = val

        myidx = 174
        xpos0 = (myidx / 25) * 29
        ypos0 = (myidx % 25) * 29
        xpos1 = xpos0 + 30
        ypos1 = ypos0 + 30
        myres[xpos0:xpos1, ypos0:ypos1, 1] = val
        myres[xpos0:xpos1, ypos0:ypos1, 2] = val

        myidx = 184
        xpos0 = (myidx / 25) * 29
        ypos0 = (myidx % 25) * 29
        xpos1 = xpos0+30
        ypos1 = ypos0+30
        myres[xpos0:xpos1, ypos0:ypos1, 1] = val
        myres[xpos0:xpos1, ypos0:ypos1, 2] = val


        VIS.imsave('mnist_200_intrinsic_rec.pdf', myres)

    elif dataset=='CIFAR':

        myres = numx.ones((rec_data.shape[0] + 2, rec_data.shape[1] + 2, 3))
        myres[1:rec_data.shape[0] + 1, 1:rec_data.shape[1] + 1, 0] = rec_data
        myres[1:rec_data.shape[0] + 1, 1:rec_data.shape[1] + 1, 1] = rec_data
        myres[1:rec_data.shape[0] + 1, 1:rec_data.shape[1] + 1, 2] = rec_data
        val = 255

        myidx = 41
        xpos0 = (myidx / 25) * 33
        ypos0 = (myidx % 25) * 33
        xpos1 = xpos0 + 34
        ypos1 = ypos0 + 34
        myres[xpos0:xpos1, ypos0:ypos1, 2] = val
        myres[xpos0:xpos1, ypos0:ypos1, 0] = val

        myidx = 48
        xpos0 = (myidx / 25) * 33
        ypos0 = (myidx % 25) * 33
        xpos1 = xpos0 + 34
        ypos1 = ypos0 + 34
        myres[xpos0:xpos1, ypos0:ypos1, 2] = val
        myres[xpos0:xpos1, ypos0:ypos1, 0] = val

        myidx = 108
        xpos0 = (myidx / 25) * 33
        ypos0 = (myidx % 25) * 33
        xpos1 = xpos0 + 34
        ypos1 = ypos0 + 34
        myres[xpos0:xpos1, ypos0:ypos1, 2] = val
        myres[xpos0:xpos1, ypos0:ypos1, 0] = val

        myidx = 194
        xpos0 = (myidx / 25) * 33
        ypos0 = (myidx % 25) * 33
        xpos1 = xpos0 + 34
        ypos1 = ypos0 + 34
        myres[xpos0:xpos1, ypos0:ypos1, 2] = val
        myres[xpos0:xpos1, ypos0:ypos1, 1] = val

        #VIS.plot([xpos, xpos+33, xpos+33, xpos,xpos], [ypos, ypos, ypos+33, ypos+33,ypos],lw=1, color='red')


        VIS.imsave('cifar_200_intrinsic_rec.pdf',myres)
        #VIS.plot([xpos, xpos], [ypos + 33, ypos], lw=2, color='red')
        #VIS.plot([xpos + 33, ypos + 33], [xpos , ypos + 33], lw=2, color='green')
        #VIS.plot([xpos + 33, ypos + 33], [xpos + 33, ypos ], lw=2, color='green')

        #IO.save_image(myres,'cifar_200_intrinsic_rec','png')


    def show_reconstruction(idx):
        longS =130
        if dataset == 'CIFAR':
            longS += 4*5
        num_pattern= 15
        cueEC = EC_data_train[idx]

        cueMSI = ae.decode(cueEC).reshape(d1, d2)

        target = ae.decode(numx.roll(EC_data_train, -idx - 1, 0))[0:num_pattern, :]

        corr = BinaryNoise(0.1)
        cueEC_noiseEC10 = corr.corrupt(EC_data_train[idx])
        cueMSI_noiseEC10 = ae.decode(cueEC_noiseEC10).reshape(d1, d2)

        corr = BinaryNoise(0.2)
        cueEC_noiseEC20 = corr.corrupt(EC_data_train[idx])
        cueMSI_noiseEC20 = ae.decode(cueEC_noiseEC20).reshape(d1, d2)

        res = ae.decode(models[0].full_intrinsic_loop_encode_next_decode(cueEC))[0:num_pattern, :]
        res_noiseEC10 = ae.decode(models[0].full_intrinsic_loop_encode_next_decode(cueEC_noiseEC10))[0:num_pattern, :]
        res_noiseEC20 = ae.decode(models[0].full_intrinsic_loop_encode_next_decode(cueEC_noiseEC20))[0:num_pattern, :]


        rec_data = VIS.tile_matrix_rows(
            matrix=res.T,
            tile_width=d1,
            tile_height=d2,
            num_tiles_x=1,
            num_tiles_y=num_pattern,
            border_size=1,
            normalized=True)
        rec_data = numx.hstack((numx.ones((d1, longS)),cueMSI, numx.ones((d1, 20)), rec_data))

        rec_data_noiseEC10 = VIS.tile_matrix_rows(
            matrix=res_noiseEC10.T,
            tile_width=d1,
            tile_height=d2,
            num_tiles_x=1,
            num_tiles_y=num_pattern,
            border_size=1,
            normalized=True)
        rec_data_noiseEC10 = numx.hstack((numx.ones((d1, longS)),cueMSI_noiseEC10, numx.ones((d1, 20)), rec_data_noiseEC10))

        rec_data_noiseEC20 = VIS.tile_matrix_rows(
            matrix=res_noiseEC20.T,
            tile_width=d1,
            tile_height=d2,
            num_tiles_x=1,
            num_tiles_y=num_pattern,
            border_size=1,
            normalized=True)
        rec_data_noiseEC20 = numx.hstack((numx.ones((d1, longS)),cueMSI_noiseEC20, numx.ones((d1, 20)), rec_data_noiseEC20))

        rec_target = VIS.tile_matrix_rows(
            matrix=target.T,
            tile_width=d1,
            tile_height=d2,
            num_tiles_x=1,
            num_tiles_y=num_pattern,
            border_size=1,
            normalized=True)
        rec_target = numx.hstack((numx.ones((d1, longS)),cueMSI, numx.ones((d1, 20)), rec_target))


        #VIS.imshow_matrix(matrix=rec,
        #                  windowtitle='Reconstruction for cue ' + str(idx) )

        VIS.figure('reconstruction_for_cue_' + str(idx+1))  # .suptitle()
        VIS.axis('off')
        VIS.gray()
        sidx = 0
        if dataset=='CIFAR':
            rec = numx.vstack((numx.ones((10, rec_data.shape[1])), rec_target, numx.ones((30, rec_data.shape[1])),
                               rec_data, numx.ones((5, rec_data.shape[1])), rec_data_noiseEC10,
                               numx.ones((5, rec_data.shape[1])), rec_data_noiseEC20))
            VIS.text(0, sidx+2, '                                                      Visualized Ground Truth', fontsize=10)
            VIS.text(0, sidx+32+32, '                                                          Recalled Sequence', fontsize=10)
            VIS.text(0, sidx+32+32, '                    Cue ('+str(idx+1)+')', fontsize=10)
            sidx += 10
        else:
            rec = numx.vstack((numx.ones((10, rec_data.shape[1])), rec_target, numx.ones((25, rec_data.shape[1])),
                               rec_data, numx.ones((5, rec_data.shape[1])), rec_data_noiseEC10,
                               numx.ones((5, rec_data.shape[1])), rec_data_noiseEC20))
            VIS.text(0, sidx+2, '                                                      Visualized Ground Truth', fontsize=10)
            VIS.text(0, sidx+28+28, '                                                          Recalled Sequence', fontsize=10)
            VIS.text(0, sidx+28+28, '                   Cue ('+str(idx+1)+')', fontsize=10)
        sidx += 10+30
        VIS.text(0, sidx, '', fontsize=10)
        sidx += d2+13
        VIS.text(0, sidx, '  0% noise EC', fontsize=10)
        sidx += d2+5
        VIS.text(0, sidx, '10% noise EC', fontsize=10)
        sidx += d2+5
        VIS.text(0, sidx, '20% noise EC', fontsize=10)

        VIS.imshow(np.array(rec, np.float64), interpolation=None)

    def show_reconstruction_test(idx, roll_idx):
        longS =130
        if dataset == 'CIFAR':
            longS += 4*5
        num_pattern=15
        cue = EC_data_test[idx]
        res = ae.decode(models[0].full_intrinsic_loop_encode_next_decode(cue))[0:num_pattern, :]
        target = ae.decode(numx.roll(EC_data_train, -roll_idx - 1, 0))
        target_cue = target[CA3_capacity-1].reshape(d1, d2)
        target = target[0:num_pattern, :]
        cue = ae.decode(cue).reshape(d1, d2)
        rec_data = VIS.tile_matrix_rows(
            matrix=res.T,
            tile_width=d1,
            tile_height=d2,
            num_tiles_x=1,
            num_tiles_y=num_pattern,
            border_size=1,
            normalized=True)
        print rec_data.shape, cue.shape
        rec_data = numx.hstack((numx.ones((d1, longS)),cue, numx.ones((d1, 20)), rec_data))
        if roll_idx < 0:
            rec_target = numx.ones((d1, rec_data.shape[1]))
        else:
            rec_target = VIS.tile_matrix_rows(
                matrix=target.T,
                tile_width=d1,
                tile_height=d2,
                num_tiles_x=1,
                num_tiles_y=num_pattern,
                border_size=1,
                normalized=True)
            rec_target = numx.hstack((numx.ones((d1, longS)),target_cue, numx.ones((d1, 20)), rec_target))


        rec = numx.vstack(
                (numx.ones((10, rec_data.shape[1])), rec_target, numx.ones((25, rec_data.shape[1])), rec_data))
        VIS.figure('reconstruction_for_test_cue_' + str(idx))  # .suptitle()
        VIS.axis('off')
        VIS.gray()
        sidx = 0
        if dataset=='CIFAR':
            VIS.text(0, sidx+2, '                                                      Visualized Ground Truth', fontsize=10)
            VIS.text(0, sidx+32+32, '                                                           Recalled Sequence', fontsize=10)
            VIS.text(0, sidx+32+32, '                    Cue ('+str(idx+1)+')', fontsize=10)
            sidx += 5
        else:
            VIS.text(0, sidx+2, '                                                      Visualized Ground Truth', fontsize=10)
            VIS.text(0, sidx+28+28, '                                                          Recalled Sequence', fontsize=10)
            VIS.text(0, sidx+28+28, '                   Cue ('+str(idx+1)+')', fontsize=10)
        #if dataset=='CIFAR':
        #    VIS.text(0, sidx, '                    Cue (test)                          Recalled Sequence', fontsize=10)
        #else:
        #    VIS.text(0, sidx, '                   Cue (test)                            Recalled Sequence', fontsize=10)
        sidx += 25
        if roll_idx < 0:
            VIS.text(0, sidx, '                        N/A                                        N/A       ', fontsize=10)
        sidx += 15
        VIS.text(0, sidx, '           ', fontsize=10)
        sidx += d2+13
        VIS.text(0, sidx, '  0% noise EC', fontsize=10)
        VIS.imshow(np.array(rec, np.float64), interpolation=None)



    show_reconstruction(idx = 0)
    show_reconstruction(idx = 3)
    show_reconstruction(idx = 8)
    show_reconstruction(idx = 23)
    show_reconstruction(idx = 3*(CA3_capacity/10)-1)
    show_reconstruction(idx = 6*(CA3_capacity/10)-1)
    show_reconstruction(idx = 9*(CA3_capacity/10)-1)

    if dataset == 'MNIST':
        show_reconstruction_test(idx = 0,  roll_idx = 184)
        show_reconstruction_test(idx = 3*(CA3_capacity/10), roll_idx = -1)
        show_reconstruction_test(idx = 6*(CA3_capacity/10),roll_idx = 192)
        show_reconstruction_test(idx = 9*(CA3_capacity/10),roll_idx = -1)

    max_corr = get_maximum_paarwise_correlation(train_data)
    print 'Max pairwise corr'
    print 'Mean of Max:',numx.mean(max_corr)
    print 'Max of Max', numx.max(max_corr), numx.argmax(max_corr)
    print max_corr

    vis.figure('Distribution of maximum pairwise correlation', figsize=(13.7, 4.8))

    x = numx.arange(max_corr.shape[0])+1
    y = max_corr

    vis.plot(x, y, linestyle='-')

    max_corr = get_maximum_paarwise_correlation(EC_data_train)
    print 'Max pairwise corr'
    print 'Mean of Max:',numx.mean(max_corr)
    print 'Max of Max', numx.max(max_corr), numx.argmax(max_corr)
    print max_corr

    y = max_corr

    vis.plot(x, y, linestyle='--')

    max_corr = get_maximum_paarwise_correlation(models[0].EC_DG.calculate_output(EC_data_train))
    print 'Max pairwise corr'
    print 'Mean of Max:',numx.mean(max_corr)
    print 'Max of Max', numx.max(max_corr), numx.argmax(max_corr)

    y = max_corr

    vis.plot(x, y, linestyle='-.')
    legend = ["MSI", "EC", "DG"]

    vis.xlabel('Pattern index ('+str(CA3_capacity)+' = latest pattern)')
    vis.ylabel('Maximal correlation')

    vis.ylim(-0.01, 1.01)
    vis.xlim(0.0,max_corr.shape[0]+1)

    vis.legend(legend, loc=0)


    if dataset == 'MNIST':
        VIS.figimage(train_data[4-1].reshape(28,28), 178, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[15 - 1].reshape(28, 28), 236, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[24-1].reshape(28,28), 284, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[41-1].reshape(28,28), 373, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[60-1].reshape(28,28), 474, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[80-1].reshape(28,28), 580, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[96-1].reshape(28,28), 664, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[103-1].reshape(28,28), 701, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[113-1].reshape(28,28), 753, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[120-1].reshape(28,28), 792, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[142-1].reshape(28,28), 907, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[170-1].reshape(28,28), 1055, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[186-1].reshape(28,28), 1138, 430, alpha=1, zorder=1)
        VIS.figimage(train_data[194-1].reshape(28,28), 1183, 430, alpha=1, zorder=1)
    VIS.draw()
    VIS.savefig('test.png')

    show_reconstruction(idx = 186-1)
    show_reconstruction(idx = 194-1)
    show_reconstruction(idx = 3-1)

VIS.show()

















