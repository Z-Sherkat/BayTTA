# BayTTA

This repository contains a TensorFlow implementation of BayTTA (Bayesian based TTA), from the paper: 

[Medical image classification with optimized test-time augmentation using
Bayesian model averaging]

by Zeinab Sherkatghanad, Moloud Abdar, Mohammadreza Bakhtyari, Vladimir Makarenkov



# Introduction

BayTTA (Bayesian based TTA) is a method that optimizes test-time augmentation using Bayesian model averaging. This technique generates a model list associated with different variations of the input data created through TTA. BMA then combines model predictions weighted by their respective posterior probabilities. Such an approach allows one to consider model uncertainty and thus enhance the predictive performance of machine learning models. 

<p align="center">

In this repo, we implement BayTTA for image classification with several different architectures on three public medical image datasets comprising skin cancer, breast cancer, and chest X-ray images as well as two popular gene editing datasets, CRISPOR and GUIDE-seq.



# Installation:

```bash
python setup.py 
```


```bash
https://github.com/......
```


See requirements.txt file for requirements that came from our setup. All experiments were run on 
Compute Canada cluster with NVIDIA Tesla P100 and NVIDIA v100 GPUs (the Cedar cluster).

# File Structure

```
|-- requirements.txt (The requirement for reproducing the environment)
|-- plots (The model and results images)
|-- setup.py (Make this project pip installation with 'pip install -e')
|-- src/ 
|   |-- __init__.py
|   |-- checkpoint/ (Checkpoints for trained model)
|   |-- constants/ (Script to specified constant values for training our models)
|   |-- dataset/ (Folder for description of datasets and the URL link for the datasets
|   |   +-- dataset_description.txt
|   |-- models/
|    |   +-- Bay-TTA.py (Used trained model for making the prediction based on BayTTA method) 
|    |   +-- BMA.py (Class definition for BMA)
|    |   +-- train_models.py (Train models)
|   |-- utils/ (Utility functions to process the data and metrics)

```


# Results

## Skin cancer dataset

Comparison of the baseline models accuracy (%) ± STD performance with TTA and BayTTA versions on the skin cancer dataset. 


|Models 	|VGG-16 	|MobileNetV2 	|DenseNet201 	|ResNet152V2 	|InceptionResNetV2|
| --------------|:-------------:|:-------------:|:-------------:|:-------------:|:---------------:|
|Baseline 	|84.95 ± 0.40	|85.75 ± 1.31	|88.17 ± 1.08	|84.24 ± 1.08	|81.63 ± 1.70	  |
|TTA 		|86.06 ± 0.21 	|87.25± 0.26 	|89.28 ± 0.25 	|84.98 ± 0.43 	|83.22± 0.35	  |
|BayTTA(ours) 	|86.22 ± 0.11	|87.73± 0.02	|89.70 ± 0.007	|85.00 ± 0.17	|83.94 ± 0.17	  |


## Breast cancer  dataset
Comparison of the baseline models accuracy (%) ± STD performance with TTA and BayTTA versions on the breast cancer dataset. 


|Models 	|VGG-16 	|MobileNetV2 	|DenseNet201 	|ResNet152V2 	|InceptionResNetV2|
| --------------|:-------------:|:-------------:|:-------------:|:-------------:|:---------------:|
|Baseline 	|88.92 ± 1.70 	|86.70 ± 0.94	| 85.42 ± 2.10	| 91.52 ± 1.18	| 91.25 ± 0.98	  |
|TTA 		|90.33 ± 0.19 	|88.23 ± 0.86	| 90.09 ± 0.90	| 92.47 ± 0.51	| 92.95 ± 0.50	  |
|BayTTA(ours) 	|90.44 ± 0.04	| 88.78 ± 0.08	| 87.44 ± 1.23	| 92.89 ± 0.14*	| 92.55 ± 0.29	  |



## Chest X-ray dataset
Comparison of the baseline models accuracy (%) and STD performance with TTA and BayTTA versions on the chest X-ray dataset.


|Models 	|VGG-16 	|MobileNetV2 	|DenseNet201 	|ResNet152V2 	|InceptionResNetV2|
| --------------|:-------------:|:-------------:|:-------------:|:-------------:|:---------------:|
|Baseline 	|71.02 ± 1.01 	| 74.59 ± 1.04 	| 66.77 ± 1.11 	| 62.45 ± 0.17 	|70.45 ± 1.80	  |
|TTA  		|72.73 ± 0.33 	| 75.45 ± 0.39 	| 68.40 ± 0.15 	| 62.47 ± 0.05 	|71.61± 0.41	  |
|BayTTA(ours)  	|71.95 ± 0.17  	|75.48 ± 0.06* 	| 68.91 ± 0.12 	| 62.50 ± 0.004 |73.08± 0.23	  |


## Gene editing datasets
Accuracy results of our proposed method on the CRISPOR and GUIDE-seq datasets.

|Models 	|CRISPOR 	|GUIDE-seq	|
|CNN		| 99.82 ± 0.02	| 90.72 ± 0.36  |
|CNN+TTA	| 99.86 ± 0.008	| 91.35 ± 0.35  |
|CNN+BayTTA 	|99.87± 0.008	| 91.73 ± 0.16  |






