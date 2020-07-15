# Data Mining Lab 2020: The Variational Fair Autoencoder
 
This repository contains a pytorch (+ pytorch_lightning) implementation of the **Variational Fair Autoencoder** (VFAE) 
as proposed in ["The Variational Fair Autoencoder"](https://arxiv.org/abs/1511.00830). The code was written in order to 
reproduce the original paper's results and to experiment with further ideas.   

The VFAE is a variant of the [Variational Autoencoder](https://arxiv.org/abs/1312.6114) neural network architecture 
with the goal of producing **fair** latent representations. In the fair classification setting, input features contain a 
so called **sensible** or **protected** feature **s** that indicates membership to a protected (e.g. minority-) class. 
This could for example be gender, religion or age.  
The VFAE tries to produce latent representations of the inputs that no longer contain sensible information about **s**
while staying useful for downstream tasks like classification.

## Requirements
The code has been tested with **Python 3.7**. All requirements are specified in `environment.yml`. If you're using 
[anaconda](https://www.anaconda.com/) you can easily create an environment that meets the requirements by running:

```shell script
$ conda env create -f environment.yml
```
Otherwise you can also install the dependencies manually using `pip3`. If you're training on cpu exclusively, 
`cudatoolkit` is not required. 

## Training
If you want to run training on the [Adult Income Dataset](http://archive.ics.uci.edu/ml/datasets/Adult), first download 
the data by running:
```shell script
$ python get_data.py 
```
The script will download and pre-process the data according to the configuration in `conf/data_prepare.yaml`.

Then you can start the training by running:
 ```shell script
$ python train.py 
```   

For setting hyperparameters see

## Project structure

The project uses [hydra](https://hydra.cc/) for managing configuration, including pre-processing and training 
hyperparameters. It allows for a modular configuration composed of individual configuration files which can also be 
overwritten via command line arguments. 

You can find global settings regarding training and testing in `conf/config.yaml`. For model specific configuration, a 
yaml file is placed in `conf/model/` for each model.
Settings can also be passed as command line arguments e.g. :
```shell script
$ python train.py training.batch_size=32 dataset.predict_s=True model.params.z_dim=100
``` 

Structure overview:
```
# hydra config files for composing a global configuration
├── conf
    # main configuration for training and testing
│   ├── config.yaml
   
    # configuration for data preparation
│   ├── data_prepare.yaml
   
    # (pytorch) dataset specification
│   ├── dataset
            ...
   
    # model specification
│   └── model
            ...

# default folder where data is stored when using data_prepare
├── data

# pre-processing scripts for each dataset
├── data_prepare
│   ├── adult.py
│   └── base_prepare.py

# dataset classes for loading pytorch data loading
├── data_source
│   └── adult.py

# run download and pre-processing
├── get_data.py

# this is were the training/testing loop is defined
├── lightning_wrapper.py

# custom losses and metrics
├── losses.py
├── metrics.py

# all pytorch models are here
├── model
│   ├── logistic_regression.py
│   └── vfae.py

# main scripts for training and testing
├── test.py
├── train.py

# script for visualization of the dataset
└── visualize.py
```



## Dataset
We provide code for preparing the [Adult Income Dataset](http://archive.ics.uci.edu/ml/datasets/Adult) for training. The 
dataset consists of 45222 instances with 15 features each where an instance represents a person. Each person is 
associated with a binary label that indicates whether the person has a yearly income that is greater than 50k. The 
protected attribute is gender with female being the protected class.

We pre-process the dataset by one-hot encoding each feature. For numerical features, by default, we generate categorical 
features that are bucketized into 5 equally large ranges. However, this can be changed in `conf/data_prepare.yaml` 
before running the pre-processing script.


