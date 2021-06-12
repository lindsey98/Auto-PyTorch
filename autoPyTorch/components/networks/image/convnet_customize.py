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
        x = x.reshape(x.size(0), -1, 1, 1)
        x = self.last_layers(x)
        x = x.reshape(x.size(0), -1)
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x

    def _build_net(self, out_features):
        layers = list()
        init_filter = self.config["conv_init_filters"]
        second_filter = self.config["conv_second_filters"]
        third_filter = self.config["conv_third_filters"]
        
        # 3 CNN layers
        self._add_layer(layers, self.channels, init_filter, 1)
        cw, ch = self._get_layer_size(self.iw, self.ih)
        self.dense_size = init_filter * cw * ch # 8 x 16 x 16
        
        cw, ch = self._get_layer_size(cw, ch)
        self._add_layer(layers, init_filter, second_filter, 2)
        self.dense_size = second_filter * cw * ch # 8 x 8 x 8
        
        cw, ch = self._get_layer_size(cw, ch)
        self._add_layer(layers, second_filter, third_filter, 3)
        self.dense_size = third_filter * cw * ch # 8 x 4 x 4
        print(self.dense_size)
            
        # 2 dense layers
        dense_layers = list()
        self._add_dense(dense_layers, self.dense_size, 32, True)
        self._add_dense(dense_layers, 32, out_features, False) # last layer shall have no activation

        nw = nn.Sequential(*layers)
        nd = nn.Sequential(*dense_layers)
        print(nw, nd)
        return nw, nd
    
    def _get_layer_size(self, w, h):
        cw = w
        ch = h
        cw, ch = cw // 2, ch // 2
        return cw, ch

    def _add_layer(self, layers, in_filters, out_filters, layer_id):
        layers.append(nn.Conv2d(in_filters, out_filters,
                                kernel_size=3,
                                stride=1,
                                padding=1))
        layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2))

        
    def _add_dense(self, layers, in_filters, out_filters, activate=True):
        layers.append(nn.Conv2d(in_filters, out_filters, # write FC layer in Convolutional way
                                kernel_size=1, 
                                bias=True))
        if activate:
            layers.append(nn.ReLU(inplace=True))    
        
    @staticmethod
    def get_config_space(conv_init_filters=[8, 64], conv_second_filters=[8, 64], conv_third_filters=[8, 64]):
        cs = CS.ConfigurationSpace()
        
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_init_filters', lower=8, upper=64))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_second_filters', lower=8, upper=64))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('conv_third_filters', lower=8, upper=64))

        return(cs)
# 