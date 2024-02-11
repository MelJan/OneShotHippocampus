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

# Pretrained Layers

The different pretrained Layers can be downloaded here

## Small network

[MNIST-AutoEncoder->EC](https://drive.google.com/file/d/1BI8rTaaL-UA4UlJxP4VquVlzovvWYbjp/view?usp=drive_link)

[EC->DG](https://drive.google.com/file/d/1csN-WFMVURDE87IxpF3amFPPEVCLFQ9E/view?usp=drive_link)

[CA3->CA3](https://drive.google.com/file/d/1ODoU8Gh2jgfUZhvR7HGuy_c9A_rli0Nn/view?usp=drive_link)

## Large network 

[MNIST-AutoEncoder->EC](https://drive.google.com/file/d/12c3xmLX7U1ZAyCMSjnhqUi6z0aPMhCLZ/view?usp=drive_link)

[EC->DG](https://drive.google.com/file/d/1ip0t3z1_YDmnshtKYSFyIPPhpc1F-CwG/view?usp=drive_link)

[CA3->CA3-activity3.2%](https://drive.google.com/file/d/1_ogsamCGoIm1-g3sY-CV_pUO3ztWTSsA/view?usp=drive_link)

[CA3->CA3-activity10.0%](https://drive.google.com/file/d/13NmtlUCit8e372Eo6fPOKm7sogt5WWD2/view?usp=drive_link)

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
