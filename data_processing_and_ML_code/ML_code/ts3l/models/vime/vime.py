from typing import Tuple

import torch
import torch.nn as nn

from ts3l.models.common import TS3LModule
from .vime_self import VIMESelfSupervised
from .vime_semi import VIMESemiSupervised_1, VIMESemiSupervised_2


class VIME(TS3LModule):
    def __init__(self, 
                input_dim: int, 
                hidden_dim: int, 
                output_dim: int,
                encoder_depth=4,
                #head_depth=2,
                dropout_rate = 0.04
                ):
        """Initialize VIME

        Args:
            input_dim (int): The dimension of the encoder
            hidden_dim (int): The hidden dimension of the predictor
            output_dim (int): The output dimension of the predictor
        """
        super(VIME, self).__init__()
        # Akhila added this 
        # VIMESelfSupervised in the original did not accept the input parameter hidden_dim
        #self.__encoder = VIMESelfSupervised(input_dim)
        self.__encoder = VIMESelfSupervised(input_dim, hidden_dim, encoder_depth, dropout_rate)
        #self.predictor = VIMESemiSupervised(input_dim, hidden_dim, output_dim)
        self.one_layer_prediction_head = VIMESemiSupervised_1(input_dim, hidden_dim, output_dim)
        self.two_layer_prediction_head = VIMESemiSupervised_2(input_dim, hidden_dim, output_dim)
        
    @property
    def encoder(self) -> nn.Module:
        return self.__encoder
        
    def _first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The first phase step of VIME

        Args:
            x (torch.Tensor): The input batch

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted mask vector and predicted features
        """
        mask_output, feature_output = self.encoder(x)
        return mask_output, feature_output
    
    
    def _second_phase_step(self, x: torch.Tensor) -> torch.Tensor:
        """The second phase step of VIME

        Args:
            x (torch.Tensor): The input batch.

        Returns:
            torch.Tensor: The predicted logits of VIME
        """
        # Akhila. This error popped up after I changed the encoder. No idea why though. 
        #print(x.shape)
        x = x.squeeze(1)
        #print(x.shape)
        
        x = self.encoder.encoder(x)
        
        if self.pred_head_size == 1:
            logits = self.one_layer_prediction_head(x)
            #print('using 1 layer head')
        else:
            logits = self.two_layer_prediction_head(x)
            #print('using 2 layer head')
        
        #logits = self.predictor(x)
        return logits