# OneShotHippocampus
This repository contains the code for the publication 

A Hippocampus Model for Online One-Shot Storage of Pattern Sequences 

https://arxiv.org/abs/1905.12937


# Create the environment

Conda: 

<code>conda env create -f environment.yml
</code>

Pip:

<code>
pip install -r requirements.txt </code>(in python2.7 env)

# Run experiments:

To run the experiments of the model with and without DG and creating the plots:

1. Choose model size f, 

    <code>f = 2 (intrinsic sequence length 200, Figure 14)</code>

    <code>f = 10 (intrinsic sequence length 1000, Figure 10, 11, 14)</code>

2. Choose a dataset out of # Chose a dataset of 

    <code>UNCORR (Figure 11 b,d,f, Figure 14 a, b)</code>

    <code>CROSSCORR (Figure 10 b,d,e, Figure 11 a,c,e, Figure 12 e, f)</code>

    <code>CORR</code>

    <code>MNIST (Figure 14 c, d)</code> 

    <code>CIFAR</code>

2. Choose  activity level 

    <code>CA3_activity = 0.2 (Figure 10 & 11)</code>

    <code>CA3_activity = 0.1 (Figure 14 a,c,e)</code>
    
    <code>CA3_activity = 0.032 (Figure 14 b,d,f)</code>

Run either 

<code>
python EC_CA3_EC.py or
</code>

<code> 
python EC_DG_CA3_EC.py
</code>