from typing import OrderedDict, Tuple

import torch
import torch.nn as nn


from ts3l.models.common import TS3LModule
from ts3l.models.common import MLP
#from ts3l.models.common import one_layer_prediction_head
#from ts3l.models.common import two_layer_prediction_head


class DAE(TS3LModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        encoder_depth=4,
        head_depth=2,
        dropout_rate = 0.04,
        output_dim = 2,
    ):
        """Implementation of Denoising AutoEncoder.
        DAE processes input data that has been partially corrupted, producing clean data during the self-supervised learning stage. 
        The denoising task enables the model to learn the input distribution and generate latent representations that are robust to corruption. 
        These latent representations can be utilized for a variety of downstream tasks.
        Args:
            input_dim (int): The size of the inputs
            hidden_dim (int): The dimension of the hidden layers
            encoder_depth (int, optional): The number of layers of the encoder MLP. Defaults to 4.
            head_depth (int, optional): The number of layers of the pretraining head. Defaults to 2.
            dropout_rate (float, optional): The probability of setting the outputs of the dropout layer to zero during training. Defaults to 0.04.
            output_dim (int, 2): The size of the outputs
        """
        super(DAE, self).__init__()

        # Akhila added this to disable batchnorm. I need to make this settable from the Config Later.  
        batchnorm = False
        
        self.__encoder = MLP(input_dim, hidden_dim, encoder_depth, dropout_rate, batchnorm)
        self.mask_predictor_head = MLP(hidden_dim, input_dim, head_depth, dropout_rate, batchnorm)
        self.reconstruction_head = MLP(hidden_dim, input_dim, head_depth, dropout_rate, batchnorm)

        # Original
        #self.head = nn.Sequential(
        #    OrderedDict([
        #        ("head_activation", nn.ReLU(inplace=True)),
        #        ("head_batchnorm", nn.BatchNorm1d(hidden_dim)),
        #        ("head_dropout", nn.Dropout(dropout_rate)),bbc
        
        #        ("head_linear", nn.Linear(hidden_dim, output_dim))
        #    ])
        #)
        
        # Akhila # single hidden layer in head
        self.one_layer_prediction_head = nn.Sequential(
            OrderedDict([
                ("head_linear_hid", nn.Linear(hidden_dim, hidden_dim)),
                #("head_batchnorm", nn.BatchNorm1d(hidden_dim)),
                ("head_activation", nn.ReLU(inplace=True)),
                #("head_dropout", nn.Dropout(0.1)),
                ("head_linear_out", nn.Linear(hidden_dim, output_dim))
            ])
        )
        
        # Akhila # 2 hidden layers in head
        self.two_layer_prediction_head = nn.Sequential(
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
    
    @property
    def encoder(self) -> nn.Module:
        return self.__encoder

    def _first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        mask = torch.sigmoid(self.mask_predictor_head(emb))
        feature = self.reconstruction_head(emb)

        return mask, feature

    # Akhila
    def _second_phase_step(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)

        if self.pred_head_size == 1:
            output = self.one_layer_prediction_head(emb)
            
        else:
            output = self.two_layer_prediction_head(emb)
            
            #output = self.head(emb)
        return output
