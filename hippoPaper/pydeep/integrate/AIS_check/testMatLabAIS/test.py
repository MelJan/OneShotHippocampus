import os
os.nice(19)

import numpy as numx
import pydeep.rbm.model as MODEL
import pydeep.rbm.trainer as TRAINER
import pydeep.rbm.estimator as ESTIMATOR
import pydeep.misc.visualization as VISUALIZATION
import pydeep.misc.io as IO
import pydeep.misc.numpyextension as npExt
import mkl
import scipy.io as sio
mkl.set_num_threads(4)

def getbasebias(data=None):
    baseBias = numx.zeros((1,784))
    if data != None:
        p_int = (data.sum(axis=0).reshape(1,data.shape[1])+5.0*600)/(100*600+5.0*600)
        baseBias = numx.log( p_int) - numx.log(1-p_int)
    return baseBias

def getbasePartition(baseBias,numhids):
    return numx.sum(numx.log(1.0+numx.exp(baseBias))) + (numhids)*numx.log(2.0);

def getbaseSamples(baseBias,num_chains):
    samples = numx.tile(1.0/(1.0 + numx.exp(-baseBias)), (num_chains, 1))
    return samples > numx.random.rand(samples.shape[0],samples.shape[1])

def getUnnormalizedProbabilityXBase(vis_states,baseBias,numhids):
    return (vis_states*baseBias.T + numhids*numx.log(2.0))


def annealed_importance_samplingBin(rbm, 
                                    num_runs, 
                                    beta,
                                    data):
    
    # Generate base bias from data
    visbiases_base = getbasebias(data)
    
    # setup temp vars
    visbias_base = numx.tile(visbiases_base.T,num_runs).T
    hidbias = numx.tile(rbm.bh.T,num_runs).T
    visbias = numx.tile(rbm.bv.T,num_runs).T

    #### Sample from the base-rate model ####
    negdata = getbaseSamples(visbiases_base,num_runs)
    
    # calculate base rate u
    logww  = -(numx.dot(negdata,visbiases_base.T) + rbm.output_dim*numx.log(2.0))

    # Calculate u
    Wh = numx.dot(negdata,rbm.w) + hidbias 
    Bv_base = numx.dot(negdata,visbiases_base.T)
    Bv = numx.dot(negdata,rbm.bv.T)  

    ### The CORE of an AIS RUN ####
    for bb in beta[1:beta.shape[0]]:  # muss bis 1.0 gehen bei ruslan nicht korrekt
        expWh = numx.exp(bb*Wh)
        logww  += (1.0-bb)*Bv_base + bb*Bv + numx.sum(numx.log(1.0+expWh),1).reshape(expWh.shape[0],1)
        
        # sample
        poshidprobs = expWh/(1.0 + expWh)
        poshidstates = poshidprobs  > numx.random.rand(num_runs,rbm.output_dim)
        negdata = 1.0/(1.0 + numx.exp(-(1.0-bb)*visbias_base - bb*(numx.dot(poshidstates,rbm.w.T) + visbias)))
        negdata = negdata > numx.random.rand(num_runs,rbm.input_dim)

        # Calculate u
        Wh      = numx.dot(negdata,rbm.w) + hidbias
        Bv_base = numx.dot(negdata,visbiases_base.T)
        Bv      = numx.dot(negdata,rbm.bv.T)
        expWh = numx.exp(bb*Wh)
        logww  -= ((1.0-bb)*Bv_base + bb*Bv + numx.sum(numx.log(1.0+expWh),1).reshape(expWh.shape[0],1))

    expWh = numx.exp(Wh)
    logww  += numx.dot(negdata,rbm.bv.T) + numx.sum(numx.log(1.0+expWh),1).reshape(expWh.shape[0],1)

    ### Compute an estimate of logZZ_est +/- 3 standard deviations. 
    r_AIS = npExt.log_sum_exp(logww) -  numx.log(num_runs)   

    # Variance berechnen
    aa = numx.mean(logww) 
    logstd_AIS = numx.log(numx.std(numx.exp ( logww-aa))) + aa - numx.log(num_runs)/2.0   

    logZZ_base = getbasePartition(visbiases_base,rbm.output_dim)
    logZZ_est = r_AIS + logZZ_base

    val = numx.vstack((numx.log(3.0)+logstd_AIS,r_AIS))
    logZZ_est_up = npExt.log_sum_exp(val) + logZZ_base
    logZZ_est_down = npExt.log_diff_exp(val) + logZZ_base
    return logZZ_est, logZZ_est_up, logZZ_est_down



#w = IO.load_matlab_file('vishid'+str(10)+'.mat', 'vishid')
#print numx.max(w)
#VISUALIZATION.imshow_matrix(w[0].reshape(28,28),'Samples')
#VISUALIZATION.show()



# Load data
loaded = IO.load_matlab_file('../../../workspacePy/data/train_images.mat', 'batchdata')
train_data = loaded[:,:,0]
for i in range(1,100):
    train_data = numx.vstack((train_data,loaded[:,:,i]))
loaded = IO.load_matlab_file('../../../workspacePy/data/test.mat', 'testbatchdata')
test_data = loaded[:,:,0]
for i in range(1,100):
    test_data = numx.vstack((test_data,loaded[:,:,i]))

# train_images model
method = 'Normal'
trial = 1
trainMethod = 'CD'
epsilon = 0.01
regL2Norm = 0.0
    
#rbm = IO.load_object(method+'_'+trainMethod+'_'+str(trial)+'_'+str(10)+'_'+str(epsilon)+'_'+str(regL2Norm)+'.rbm')
rbm = IO.load_object('test16h.rbm')
#rbm = MODEL.BinaryBinaryRBM(number_visibles = 28*28, 
#                                   number_hiddens = 500, 
#                                   data=None, 
#                                   initial_weights=0.0, 
#                                   initial_visible_bias=0.0, 
#                                   initial_hidden_bias=0.0, 
#                                   initial_visible_offsets=0.0, 
#                                   initial_hidden_offsets=0.0)

#rbm.w = IO.load_matlab_file('vishid'+str(60)+'.mat', 'vishid').T
#rbm.bh = IO.load_matlab_file('hidbiases'+str(60)+'.mat', 'hidbiases').T
#rbm.bv = IO.load_matlab_file('visbiases'+str(60)+'.mat', 'visbiases').T

## evaluate model
a = numx.linspace(0.0, 0.5, 1+1)
a = a[0:a.shape[0]-1]
b = numx.linspace(0.5, 0.9, 1+1)
b = b[0:b.shape[0]-1]
c = numx.linspace(0.0, 1.0, 1000+1)
print c
betas = numx.hstack((a,b,c))
#
numx.random.seed(42)
logZ = annealed_importance_samplingBin(rbm, num_runs=100, beta=c,data = train_data)[0][0]
print 'AIS min LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ, train_data))
numx.random.seed(42)
logZ, logZmin, logZmax, logZall = ESTIMATOR.annealed_importance_sampling(rbm,k = 1, betas = c, status = False)
print 'AIS min LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ, train_data))
logZ = ESTIMATOR.partition_function_factorize_h(rbm,1.0,12)
print 'AIS min LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ, train_data))
exit()                                
#logZstd = numx.std(logZall)
#print 'avg Z', logZ
#print 'min Z', logZmin
#print 'max Z', logZmax
#print 'std Z', logZstd
#print 'AIS min LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZmin, train_data))
#print 'AIS max LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZmax, train_data))
#print 'AIS avg LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ, train_data))
#print 'AIS avg-std LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ-logZstd, train_data))
#print 'AIS avg+std LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ+logZstd, train_data))
#
#print 'AIS min LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZmin, test_data))
#print 'AIS max LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZmax, test_data))
#print 'AIS avg LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ, test_data))
#print 'AIS avg-std LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ-logZstd, test_data))
#print 'AIS avg+std LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ+logZstd, test_data))
#    
#    
#exit()

print method
print trainMethod
print trial
print regL2Norm
print epsilon

epochs = 101
step = 10
    
v1 = 28
v2 = 28
h1 = 25
h2 = 20

batch_size = 100
uvo = 0.0
uho = 0.0
if method is 'Centered':
    rbm = MODEL.BinaryBinaryRBM(number_visibles = v1*v2, 
                                number_hiddens = h1*h2, 
                                data=train_data, 
                                initial_weights=0.01, 
                                initial_visible_bias=0.0, 
                                initial_hidden_bias=0.0, 
                                initial_visible_offsets='AUTO', 
                                initial_hidden_offsets='AUTO')
    uho = 0.001
else:
    if method is 'DataNorm':
        rbm = MODEL.BinaryBinaryRBM(number_visibles = v1*v2, 
                                    number_hiddens = h1*h2, 
                                    data=train_data, 
                                    initial_weights=0.01, 
                                    initial_visible_bias=0.0, 
                                    initial_hidden_bias=0.0, 
                                    initial_visible_offsets='AUTO', 
                                    initial_hidden_offsets=0.0)
        uho = 0.0
    else:
        rbm = MODEL.BinaryBinaryRBM(number_visibles = v1*v2, 
                                    number_hiddens = h1*h2, 
                                    data=train_data, 
                                    initial_weights=0.01, 
                                    initial_visible_bias=0.0, 
                                    initial_hidden_bias=0.0, 
                                    initial_visible_offsets=0.0, 
                                    initial_hidden_offsets=0.0)  
        uho = 0.0 

trainer =  None
k = 1
if trainMethod is 'PT':
    trainer = TRAINER.PT(rbm,20)
if trainMethod is 'CD':
    trainer = TRAINER.CD(rbm)
if trainMethod is 'PCD':
    trainer = TRAINER.PCD(rbm,batch_size)
if trainMethod is 'PCD10':
    trainer = TRAINER.PCD(rbm,batch_size)
    k = 10

IO.save_object(rbm,method+'_'+trainMethod+'_'+str(trial)+'_'+str(0)+'_'+str(epsilon)+'_'+str(regL2Norm)+'.rbm')
for epoch in range(1,epochs+1) :
     
    for b in range(0,train_data.shape[0],batch_size):
        batch = train_data[b:b+batch_size,:]
        trainer.train(data=batch,
                      num_epochs=1,
                      epsilon=epsilon, 
                      k=k,  
                      momentum=0.0, 
                      regL1Norm = 0.0,
                      regL2Norm = regL2Norm,
                      regSparseness = 0.0,
                      desired_sparseness = None,
                      update_visible_offsets = uvo,
                      update_hidden_offsets = uho,
                      offset_typ = 'DD',
                      use_centered_gradient = False,
                      restrict_gradient = None,
                      restriction_norm = 'Mat', 
                      use_hidden_states = False)

    #RE = numx.mean(ESTIMATOR.reconstruction_error(trainer.model, train_data))
    #print '%d\t%8.6f\t' % (epoch, RE)
        
    if epoch % step == 0 :
        IO.save_object(rbm,method+'_'+trainMethod+'_'+str(trial)+'_'+str(epoch)+'_'+str(epsilon)+'_'+str(regL2Norm)+'.rbm')
        # save model
        sio.savemat('vishid'+str(epoch)+'.mat', {'vishid':rbm.w})
        sio.savemat('hidbiases'+str(epoch)+'.mat', {'hidbiases':rbm.bh})
        sio.savemat('visbiases'+str(epoch)+'.mat', {'visbiases':rbm.bv})

        # evaluate model
        a = numx.linspace(0.0, 0.5, 500+1)
        a = a[0:a.shape[0]-1]
        b = numx.linspace(0.5, 0.9, 4000+1)
        b = b[0:b.shape[0]-1]
        c = numx.linspace(0.9, 1.0, 10000)
        betas = numx.hstack((a,b,c))

        logZ, logZmin, logZmax, logZall = ESTIMATOR.annealed_importance_sampling(rbm,k = 1, betas = betas, status = False)

        logZstd = numx.std(logZall)
        #print 'avg Z', logZ
        #print 'min Z', logZmin
        #print 'max Z', logZmax
        #print 'std Z', logZstd
        #print 'AIS min LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZmin, train_data))
        #print 'AIS max LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZmax, train_data))
        print 'AIS avg LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ, train_data))
        #print 'AIS avg-std LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ-logZstd, train_data))
        #print 'AIS avg+std LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ+logZstd, train_data))

        #print 'AIS min LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZmin, test_data))
        #print 'AIS max LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZmax, test_data))
        print 'AIS avg LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ, test_data))
        #print 'AIS avg-std LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ-logZstd, test_data))
        #print 'AIS avg+std LL', numx.mean(ESTIMATOR.log_likelihood_v(rbm, logZ+logZstd, test_data))