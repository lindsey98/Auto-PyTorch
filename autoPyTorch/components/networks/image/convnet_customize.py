#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customized Implementation of a convolutional network.
"""

from __future__ import division, print_function

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import torch.nn as nn

from autoPyTorch.components.networks.base_net import BaseImageNet

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


class ConvCusNet(BaseImageNet):
    def __init__(self, config, in_features, out_features, final_activation, *args, **kwargs):
        super(ConvCusNet, self).__init__(config, in_features, out_features, final_activation)
        self.layers, self.last_layers = self._build_net(self.n_classes)


    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.last_layers(x)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x

    def _build_net(self, out_features):
        layers = list()
        init_filter = self.config["conv_init_filters"]
        second_filter = self.config["conv_second_filters"]
        third_filter = self.config["conv_third_filters"]
        dense_filter = self.config["fc_first_size"]
        
        # 3 CNN layers
        self._add_layer(layers, self.channels, init_filter, 1)
        cw, ch = self._get_layer_size(self.iw, self.ih)
        self.dense_size = init_filter * cw * ch # 8 x 16 x 16
        
        cw, ch = self._get_layer_size(cw, ch)
        self._add_layer(layers, init_filter, second_filter, 2)
        self.dense_size = second_filter * cw * ch # 8 x 8 x 8
        
        cw, ch = self._get_layer_size(cw, ch)
        self._add_layer(layers, second_filter, third_filter, 3)
        self.dense_size = second_filter * cw * ch # 8 x 4 x 4
        print(cw, ch, self.dense_size)
            
        # 2 dense layers
        dense_layers = list()
        self._add_dense(dense_layers, self.dense_size, dense_filter, 4, True)
        self._add_dense(dense_layers, dense_filter, out_features, 5, False) # last layer shall have no activation

        nw = nn.Sequential(*layers)
        nd = nn.Sequential(*dense_layers)
        print(nw, nd)
        return nw, nd
    
    def _get_layer_size(self, w, h):
        cw = ((w - self.config["conv_kernel_size"] + 2 * self.config["conv_kernel_padding"])
                //self.config["conv_kernel_stride"]) + 1
        ch = ((h - self.config["conv_kernel_size"] + 2 * self.config["conv_kernel_padding"])
                //self.config["conv_kernel_stride"]) + 1
        cw, ch = cw // self.config["pool_size"], ch // self.config["pool_size"]
        return cw, ch

    def _add_layer(self, layers, in_filters, out_filters, layer_id):
        layers.append(nn.Conv2d(in_filters, out_filters,
                                kernel_size=self.config["conv_kernel_size"],
                                stride=self.config["conv_kernel_stride"],
                                padding=self.config["conv_kernel_padding"]))
        layers.append(nn.BatchNorm2d(out_filters))
        layers.append(self.activation())
        layers.append(nn.MaxPool2d(kernel_size=self.config["pool_size"], stride=self.config["pool_size"]))
    
    def _add_dense(self, layers, in_filters, out_filters, layer_id, activate=True):
        layers.append(nn.Conv2d(in_filters, out_filters, # write FC layer in Convolutional way
                                kernel_size=1, 
                                bias=True))
        if activate:
            layers.append(self.activation())    
        
    @staticmethod
    def get_config_space(user_updates=None):
        cs = CS.ConfigurationSpace()
        
        cs.add_hyperparameter(CSH.CategoricalHyperparameter('activation', ['relu'])) #'sigmoid', 'tanh',
        num_layers = CSH.UniformIntegerHyperparameter('num_layers', lower=3, upper=3) # FIXME
        cs.add_hyperparameter(num_layers)
        
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_init_filters', lower=8, upper=64))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_second_filters', lower=8, upper=64))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_third_filters', lower=8, upper=64))
        
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_kernel_size', lower=3, upper=3))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_kernel_stride', lower=1, upper=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_kernel_padding', lower=1, upper=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('pool_size', lower=2, upper=2)) # FIXME: I limit the upper to be same as lower
        
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('fc_first_size', lower=32, upper=32))

        return(cs)
# 