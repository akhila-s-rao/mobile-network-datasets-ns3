import torch
import torch.nn as nn
from typing import OrderedDict

class VIMESemiSupervised_1(nn.Module):
    def __init__(self, hidden_dim:int, output_dim:int):
        """_summary_

        Args:
            hidden_dim (int): The hidden dimension of the predictor
            output_dim (int): The output dimension of the predictor
        """
        super().__init__()

        # Akhila # single hidden layer in head
        self.prediction_head = nn.Sequential(
            OrderedDict([
                ("head_linear_hid", nn.Linear(hidden_dim, hidden_dim)),
                ("head_activation", nn.ReLU(inplace=True)),
                ("head_linear_out", nn.Linear(hidden_dim, output_dim))
            ])
        )
        

    def forward(self, x):
        """The forward pass of semi-supervised module of VIME

        Args:
            x (torch.FloatTensor): The input batch.

        Returns:
            torch.FloatTensor: The predicted logits of VIME
        """
        # Akhila single hidden layer
        logits = self.prediction_head(x)
         
        return logits


class VIMESemiSupervised_2(nn.Module):
    def __init__(self, hidden_dim:int, output_dim:int):
        """_summary_

        Args:
            hidden_dim (int): The hidden dimension of the predictor
            output_dim (int): The output dimension of the predictor
        """
        super().__init__()
        
        # Akhila # 2 hidden layers in head
        self.prediction_head = nn.Sequential(
            OrderedDict([
                ("head_linear_hid1", nn.Linear(hidden_dim, hidden_dim)),
                ("head_activation1", nn.ReLU(inplace=True)),
                ("head_linear_hid2", nn.Linear(hidden_dim, 100)),
                ("head_activation2", nn.ReLU(inplace=True)),
                ("head_linear_out", nn.Linear(100, output_dim))
            ])
        )

    def forward(self, x):
        """The forward pass of semi-supervised module of VIME

        Args:
            x (torch.FloatTensor): The input batch.

        Returns:
            torch.FloatTensor: The predicted logits of VIME
        """
        # Akhila two hidden layers
        logits = self.prediction_head(x)

        return logits
