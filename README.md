# BayTTA

This repository contains a TensorFlow implementation of BayTTA (Bayesian based TTA), from the paper: 

[Medical image classification with optimized test-time augmentation using
Bayesian model averaging]

by Zeinab Sherkatghanad, Moloud Abdar, Mohammadreza Bakhtyari, Vladimir Makarenkov



## Introduction

BayTTA (Bayesian based TTA) is a method that optimizes test-time augmentation using Bayesian model averaging. This technique generates a model list associated with different variations of the input data created through TTA. BMA then combines model predictions weighted by their respective posterior probabilities. Such an approach allows one to consider model uncertainty and thus enhance the predictive performance of machine learning models. 

[TTA-BMA-Med.pdf](https://github.com/Z-Sherkat/BayTTA/files/14213663/TTA-BMA-Med.pdf)



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
\caption{\label{tab2} Comparison of the baseline models accuracy (\%) $\pm$ STD performance with TTA and BayTTA versions on the skin cancer dataset. The asterisk denotes the superior performance of BayTTA.}
\centering
\begin{tabular}{lccccc}
\toprule
Models & VGG-16 & MobileNetV2 & DenseNet201& ResNet152V2&InceptionResNetV2\\
\hline
Baseline&84.95 $\pm$ 0.40&85.75 $\pm$ 1.31&\bf{88.17} $\pm$ 1.08&84.24 $\pm$ 1.08&81.63 $\pm$ 1.70\\ 
TTA&86.06 $\pm$ 0.21 &87.25$\pm$ 0.26 &89.28 $\pm$ 0.25 &84.98 $\pm$ 0.43 &83.22$\pm$ 0.35 \\ 
\bf{BayTTA(ours)}&\bf{86.22 $\pm$ 0.11} &\bf{87.73$\pm$ 0.02} &\bf{89.70 $\pm$ 0.007*} &\bf{85.00 $\pm$ 0.17} &\bf{83.94 $\pm$ 0.17} \\ 
\bottomrule
\end{tabular}

\bigskip 

\caption{\label{tab3} Comparison of the baseline models accuracy (\%) $\pm$ STD performance with TTA and BayTTA versions on the breast cancer dataset. The asterisk denotes the superior performance of BayTTA.}
\centering
\begin{tabular}{lccccc}
\toprule
Models & VGG-16 & MobileNetV2 & DenseNet201& ResNet152V2&InceptionResNetV2\\
\hline
Baseline&88.92 $\pm$ 1.70&86.70 $\pm$ 0.94&85.42 $\pm$ 2.10&\bf{91.52 $\pm$ 1.18}&91.25 $\pm$ 0.98\\ 
TTA&90.33 $\pm$ 0.19 &88.23 $\pm$ 0.86 &90.09 $\pm$ 0.90 &92.47 $\pm$ 0.51 &92.95 $\pm$ 0.50\\ 
\bf{BayTTA(ours)}&\bf{90.44 $\pm$ 0.04}&\bf{88.78 $\pm$ 0.08} &\bf{87.44 $\pm$ 1.23}&\bf{92.89 $\pm$ 0.14*} &\bf{92.55 $\pm$ 0.29}\\ 
\bottomrule
\end{tabular}
\bigskip
\caption{\label{tab4} Comparison of the baseline models accuracy (\%) and STD performance with TTA and BayTTA versions on the chest X-ray dataset. The asterisk denotes the superior performance of BayTTA.}
\centering
\begin{tabular}{lccccc}
\toprule
Models & VGG-16 & MobileNetV2 & DenseNet201& ResNet152V2&InceptionResNetV2\\
\hline
Baseline&71.02 $\pm$ 1.01&\bf{74.59 $\pm$ 1.04}&66.77 $\pm$ 1.11&62.45 $\pm$ 0.17&70.45 $\pm$ 1.80\\ 
TTA&72.73 $\pm$ 0.33 &75.45 $\pm$ 0.39 &68.40 $\pm$ 0.15 &62.47 $\pm$ 0.05 &71.61$\pm$ 0.41\\ 
\bf{BayTTA(ours)}&\bf{71.95 $\pm$ 0.17} &\bf{75.48 $\pm$ 0.06*} &\bf{68.91 $\pm$ 0.12}&\bf{62.50 $\pm$ 0.004} &\bf{73.08$\pm$ 0.23}\\ 
\bottomrule
\end{tabular}
\end{table*}
