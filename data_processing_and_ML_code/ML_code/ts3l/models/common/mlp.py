import torch
from torch import nn
from typing import OrderedDict

class MLP(nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, dropout=0.0, batchnorm = True):
        layers = []
        layer_dim = in_dim
        
        for i in range(n_layers - 1):
            layers.append(("linear_%d" % i, torch.nn.Linear(layer_dim, hidden_dim)))
            if batchnorm: 
                layers.append(("batchnorm_%d" % i, nn.BatchNorm1d(hidden_dim)))
            layers.append(("relu_%d" % i, nn.ReLU(inplace=True)))
            layers.append(("dropout_%d" % i, torch.nn.Dropout(dropout)))
            layer_dim = hidden_dim

        layers.append(("last_layer", torch.nn.Linear(hidden_dim, out_dim)))

        super().__init__(OrderedDict(layers))