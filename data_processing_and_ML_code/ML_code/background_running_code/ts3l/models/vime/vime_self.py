import torch
import torch.nn as nn
from ts3l.models.common import MLP

class VIMESelfSupervised(nn.Module):
    # Original 
    #def __init__(self, input_dim:int):
    # Akhila changed this so that hidden_dim would actually be used to set the size of 
    # the hidden dimentions. Before it was being used to decide the size of the predictor head's 
    # first layer. Now it is being used for both 
    def __init__(self, input_dim:int, hidden_dim:int):
    
        """Initialize self-supervised module of VIME

        Args:
            input_dim (int): The dimension of the encoder
        """
        super().__init__()
        # Original
        self.h = nn.Linear(input_dim, input_dim, bias=True)
        
        # Akhila has increased the encoder_depth
        #self.h = MLP(input_dim, hidden_dim, encoder_depth, dropout_rate)

        
        self.mask_output = nn.Linear(input_dim, input_dim, bias=True)
        self.feature_output = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x):
        """The forward pass of self-supervised module of VIME

        Args:
            x (torch.FloatTensor): The input batch.

        Returns:
            torch.FloatTensor: The predicted mask vector of VIME
            torch.FloatTensor: The predicted features of VIME
        """
        h = torch.relu(self.h(x))
        mask = torch.sigmoid(self.mask_output(h))
        feature = torch.sigmoid(self.feature_output(h))
        return mask, feature
