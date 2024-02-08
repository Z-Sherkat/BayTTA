# BayTTA

This repository contains a TensorFlow implementation of BayTTA (Bayesian based TTA), from the paper: 

[Medical image classification with optimized test-time augmentation using
Bayesian model averaging]

by Zeinab Sherkatghanad, Moloud Abdar, Mohammadreza Bakhtyari, Vladimir Makarenkov



## Introduction

BayTTA (Bayesian based TTA) is a method that optimizes test-time augmentation using Bayesian model averaging. This technique generates a model list associated with different variations of the input data created through TTA. BMA then combines model predictions weighted by their respective posterior probabilities. Such an approach allows one to consider model uncertainty and thus enhance the predictive performance of machine learning models. 

<p align="center">

In this repo, we implement BayTTA for image classification with several different architectures on three public medical image datasets comprising skin cancer, breast cancer, and chest X-ray images as well as two popular gene editing datasets, CRISPOR and GUIDE-seq.



## Installation:

```bash
python setup.py 
```


```bash
https://github.com/......
```


See requirements.txt file for requirements that came from our setup. All experiments were run on 
Compute Canada cluster with NVIDIA Tesla P100 and NVIDIA v100 GPUs (the Cedar cluster).

## File Structure

```
.
+-- requirements.txt (The requirement for reproducing the environment)
+-- plots (The model and results images)
+-- setup.py (Make this project pip installation with 'pip install -e')
+-- src/ 
|   +-- __init__.py
|   +-- checkpoint/ (Checkpoints for trained model)
|   +-- constants/ (Script to specified constant values for training our models)
|   +-- dataset/ (Folder for description of datasets and the URL link for the datasets
|   |   +-- dataset_description.txt
|   +-- models/
|    |   +-- Bay-TTA.py (Used trained model for making the prediction based on BayTTA method) 
|    |   +-- BMA.py (Class definition for BMA)
|    |   +-- train_models.py (Train models)
|   +-- utils/ (Utility functions to process the data and metrics)

```

