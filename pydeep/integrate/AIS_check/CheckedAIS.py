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
        p_int = (data.sum(axis=0).reshape(1,data.shape[1])+5.0*500)/(data.shape[0]+5.0*500)
        baseBias = numx.log( p_int) - numx.log(1-p_int)
    return baseBias

def getbasePartition(baseBias,numhids):
    return numx.sum(numx.log(1.0+numx.exp(baseBias))) + (numhids)*numx.log(2.0);

def getbaseSamples(baseBias,num_chains):
    samples = numx.tile(1.0/(1.0 + numx.exp(-baseBias)), (num_chains, 1))
    return samples > numx.random.rand(samples.shape[0],samples.shape[1])

def getUnnormalizedProbabilityXBase(vis_states,baseBias,numhids):
    return (vis_states*baseBias.T + numhids*numx.log(2.0))


def annealed_importance_sampling(rbm, 
                                    num_runs, 
                                    betas,
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
    tt=1; 

    ### The CORE of an AIS RUN ####
    for bb in betas[1:betas.shape[0]-1]:  
        tt = tt+1; 
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
