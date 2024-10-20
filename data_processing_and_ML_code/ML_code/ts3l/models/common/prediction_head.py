import torch
from torch import nn
from typing import OrderedDict

def one_layer_prediction_head(hidden_dim, output_dim):
    
    # Akhila # single hidden layer in head
    one_layer_prediction_head = nn.Sequential(
        OrderedDict([
            ("head_linear_hid", nn.Linear(hidden_dim, hidden_dim)),
            #("head_batchnorm", nn.BatchNorm1d(hidden_dim)),
            ("head_activation", nn.ReLU(inplace=True)),
            #("head_dropout", nn.Dropout(0.1)),
            ("head_linear_out", nn.Linear(hidden_dim, output_dim))
        ])
    )
    return one_layer_prediction_head
    
# Akhila # 2 hidden layers in head
def two_layer_prediction_head(hidden_dim, output_dim): 
    two_layer_prediction_head = nn.Sequential(
        OrderedDict([
            ("head_linear_hid1", nn.Linear(hidden_dim, hidden_dim)),
            #("head_batchnorm", nn.BatchNorm1d(hidden_dim)),
            ("head_activation1", nn.ReLU(inplace=True)),
            #("head_dropout", nn.Dropout(dropout_rate)),
            ("head_linear_hid2", nn.Linear(hidden_dim, 100)),
            #("head_batchnorm", nn.BatchNorm1d(hidden_dim)),
            ("head_activation2", nn.ReLU(inplace=True)),
            #("head_dropout", nn.Dropout(dropout_rate)),
            ("head_linear_out", nn.Linear(100, output_dim))
        ])
    )
    return two_layer_prediction_head