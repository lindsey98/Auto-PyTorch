# Auto-PyTorch

Copyright (C) 2021  [AutoML Groups Freiburg and Hannover](http://www.automl.org/)

## Installation

Clone repository

```sh
$ cd install/path
$ git clone https://github.com/automl/Auto-PyTorch.git
$ cd Auto-PyTorch
```
If you want to contribute to this repository switch to our current development branch

```sh
$ git checkout development
```

Install pytorch: 
https://pytorch.org/

Install Auto-PyTorch:

```sh
$ cat requirements.txt | xargs -n 1 -L 1 pip install
$ python setup.py install
```

## Project structure

```
autoPyTorch
|_ components
   |_ baselines: KNN, RF, SVM etc. ML models
   |_ ensemble: create ensemble from base models
   |_ metrics: standard metrics
   |_ networks: DL networks
   |_ optimizer: Adam/RMSprop etc.
   |_ preprocessing
   |_ regularization: mixup, shake-shake and shake-drop regularization
   |_ training
   |_ utils: configuration space
   
|_ core
   |_ api.py: Main autonet class script (define fit, refit, predict, scores logic)
   |_ autonet_classes: define default pipeline for classification, regression, image classification etc.
   |_ hpbandster_extensions: extends from https://github.com/automl/HpBandSter, hyperparamter optimization
   |_ preset: hyperparameter search space to one of tiny_cs, medium_cs or full_cs
   
|_ pipeline
   |_ base: define node
   |_ nodes: define node for Network_selector, preprocesser_selector, loss_module_selector etc.

|_ data_management: dataloader


configs: config files


```



