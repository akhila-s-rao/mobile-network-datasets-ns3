import torch
import torch.nn as nn
from typing import OrderedDict

class VIMESemiSupervised_1(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        """_summary_

        Args:
            input_dim (int): The input dimension of the predictor. Must be same to the dimension of the encoder.
            hidden_dim (int): The hidden dimension of the predictor
            output_dim (int): The output dimension of the predictor
        """
        super().__init__()

        # Akhila layers for 1 and 2 hidden layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        #self.batchNorm1 = nn.BatchNorm1d(hidden_dim)
        #self.dropout1 = torch.nn.Dropout(0.1)

    def forward(self, x):
        """The forward pass of semi-supervised module of VIME

        Args:
            x (torch.FloatTensor): The input batch.

        Returns:
            torch.FloatTensor: The predicted logits of VIME
        """
        # Akhila single hidden layer 
        x = self.fc1(x)
        #x = self.batchNorm1(x)
        x = torch.relu(x)
        #x = self.dropout1(x)
        logits = self.fc3(x)

        # Original
        #x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        #logits = self.fc3(x)
        return logits


class VIMESemiSupervised_2(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        """_summary_

        Args:
            input_dim (int): The input dimension of the predictor. Must be same to the dimension of the encoder.
            hidden_dim (int): The hidden dimension of the predictor
            output_dim (int): The output dimension of the predictor
        """
        super().__init__()
        


        # Akhila layers for 1 and 2 hidden layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 100)
        self.fc3 = nn.Linear(100, output_dim)
        
        # Akhila added this
        #self.batchNorm1 = nn.BatchNorm1d(hidden_dim)
        #self.batchNorm2 = nn.BatchNorm1d(100)
        #self.dropout1 = torch.nn.Dropout(0.1)
        #self.dropout2 = torch.nn.Dropout(0.1)

    def forward(self, x):
        """The forward pass of semi-supervised module of VIME

        Args:
            x (torch.FloatTensor): The input batch.

        Returns:
            torch.FloatTensor: The predicted logits of VIME
        """
        # Akhila two hidden layers 
        x = self.fc1(x)
        #x = self.batchNorm1(x)
        x = torch.relu(x)
        #x = self.dropout1(x)
        x = self.fc2(x)
        #x = self.batchNorm2(x)
        x = torch.relu(x)
        #x = self.dropout2(x)
        logits = self.fc3(x)

        # Original
        #x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        #logits = self.fc3(x)
        return logits
