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

def getbasebias(rbm, data=None):
    base_bv = numx.zeros((1,rbm.input_dim))
    base_bh = numx.zeros((1,rbm.output_dim))
    if data != None:
        data_sum = data.sum(axis=0).reshape(1,rbm.input_dim)
        hidden_sum = rbm.probability_h_given_v(data).sum(axis=0).reshape(1,rbm.output_dim) 
        factor = 500#data.shape[0]/100.0
        ratio_data = (data_sum+5.0*factor)/(data.shape[0]+5.0*factor)
        ratio_hidden = (hidden_sum+5.0*factor)/(data.shape[0]+5.0*factor)
        base_bh = numx.log( ratio_hidden) - numx.log(1-ratio_hidden)
        base_bv = numx.log( ratio_data) - numx.log(1-ratio_data)
    return base_bv, base_bh

def annealed_importance_sampling(rbm, 
                                    num_runs, 
                                    betas,
                                    k = 1):

    #### Sample from the base-rate model ####
    negdata = rbm.probability_v_given_h(numx.zeros((num_runs, rbm.output_dim)), 0.0,True)
    negdata = rbm.sample_v(negdata)
    
    # calculate base rate u
    u = -rbm.unnormalized_log_probability_v(negdata, 0.0, True)
    
    ### The CORE of an AIS RUN ####
    for beta in betas[1:betas.shape[0]-1]:  
        
        u += rbm.unnormalized_log_probability_v(negdata, beta, True)
        for _ in range(0, k):
            poshidprobs = rbm.probability_h_given_v(negdata, beta, True)
            poshidstates = rbm.sample_h(poshidprobs)
            negdata = rbm.probability_v_given_h(poshidstates, beta, True)
            negdata = rbm.sample_v(negdata)
            
        u  -= rbm.unnormalized_log_probability_v(negdata, beta, True)

    #expWh = numx.exp(Wh)
    u  += rbm.unnormalized_log_probability_v(negdata, 1.0,True)

    ### Compute an estimate of logZZ_est +/- 3 standard deviations. 
    r_AIS = npExt.log_sum_exp(u) -  numx.log(num_runs)   

    # Variance berechnen
    aa = numx.mean(u) 
    logstd_AIS = numx.log(numx.std(numx.exp ( u-aa))) + aa - numx.log(num_runs)/2.0   

    logZZ_base = rbm._base_log_partition()
    logZZ_est = r_AIS + logZZ_base

    val = numx.vstack((numx.log(3.0)+logstd_AIS,r_AIS))
    logZZ_est_up = npExt.log_sum_exp(val) + logZZ_base
    logZZ_est_down = npExt.log_diff_exp(val) + logZZ_base
    return logZZ_est, logZZ_est_up, logZZ_est_down

def annealed_importance_samplingOLD(rbm, 
                                    num_runs, 
                                    beta,
                                    data):
    ''' Approximates the partition function for the given model using annealed
        importance sampling.
    
    :Parameters:
        model:      The model.
                   -type: Valid RBM model.
        
        num_chains: Number of AIS runs.
                   -type: int
        
        k:          Number of Gibbs sampling steps.
                   -type: int
        
        beta:       Number or a list of inverse temperatures to sample from.
                   -type: int, numpy array [num_betas]
        
        status:     If true prints the progress on console.
                   -type: bool
        
    
    :Returns:
        Mean estimated log partition function.
       -type: float
        Minimal estimated log partition function.
       -type: float
        Maximal estimated log partition function.
       -type: float
        All estimated partition function values.
       -type: numpy array
    
    '''   
# Setup temerpatures if not given 
    if status == True:
        print "Calculating the partition function using AIS: " 
        print '%3.2f' % (0.0), '%'
        
    if numx.isscalar(betas):
        betas = numx.linspace(0.0, 1.0, betas)
    
    vis_states = numx.zeros((num_chains, model.input_dim))
    beta = 0.0
    for _ in range(0, k):
        hid_probs = model.probability_h_given_v(vis_states, beta)
        hid_states = model.sample_h(hid_probs, beta)
        vis_probs = model.probability_v_given_h(hid_states, beta)   
        vis_states = model._sample_v_AIS(vis_probs, beta)
    # calcluate unnormalized LL
    u = -model._unnormalized_log_probability_v_AIS(vis_states, beta)    
    
    # Sample from a persistent markov chain and sum the unnormalized LL 
    # for all temperatures
    max_t = len(betas)
    
    # Setup temerpatures if not given 
    if status == True:
        print '%3.2f' % (100.0*numx.double(1)/numx.double(max_t)), '%'
        
    for t in range(1, max_t-1):
        beta = betas[t]
        # calcluate unnormalized LL
        u += model._unnormalized_log_probability_v_AIS(vis_states, beta)
        for _ in range(0, k):   
            hid_probs = model.probability_h_given_v(vis_states, beta)
            hid_states = model.sample_h(hid_probs, beta)
            vis_probs = model.probability_v_given_h(hid_states, beta)
            vis_states = model._sample_v_AIS(vis_probs, beta)
        # calcluate unnormalized LL
        u -= model._unnormalized_log_probability_v_AIS(vis_states, beta) 
               
        if status == True:
            print '%3.2f' % (100.0*numx.double(t+1)/numx.double(max_t)), '%'
    
    beta = betas[max_t-1]
    # calcluate unnormalized LL
    u += model._unnormalized_log_probability_v_AIS(vis_states, beta) 
    u = numx.float128(u)
    
    return (npExt.log_sum_exp(u[:])  - numx.log(num_chains) 
            + model._base_log_partition(),
            numx.min(u)+ model._base_log_partition(), 
            numx.max(u)+ model._base_log_partition(),
            u + model._base_log_partition()) 
