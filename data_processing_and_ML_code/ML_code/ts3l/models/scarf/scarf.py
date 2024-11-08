from typing import OrderedDict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ts3l.models.common import TS3LModule
from ts3l.models.common import MLP


class SCARF(TS3LModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        encoder_depth=4,
        head_depth=2,
        dropout_rate = 0.1,
    ) -> None:
        """Implementation of SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.
        It consists in an encoder that learns the embeddings.
        It is done by minimizing the contrastive loss of a sample and a corrupted view of it.
        The corrupted view is built by replacing a random set of features by another sample randomly drawn independently.
            Args:
                input_dim (int): The size of the inputs.
                hidden_dim (int): The dimension of the hidden layers.
                output_dim (int): The dimension of output.
                encoder_depth (int, optional): The number of layers of the encoder MLP. Defaults to 4.
                head_depth (int, optional): The number of layers of the pretraining head. Defaults to 2.
                dropout_rate (float, optional): A hyperparameter that is to control dropout layer. Default is 0.04.
        """
        super(SCARF, self).__init__()
        
        self.__encoder = MLP(input_dim, hidden_dim, hidden_dim, encoder_depth, dropout_rate, batchnorm=False)
        # No batchnorm or dropout for these part
        self.pretraining_head = MLP(hidden_dim, hidden_dim, hidden_dim, head_depth, dropout=0.0, batchnorm=False)
        
        # Akhila # single hidden layer in head
        self.one_layer_prediction_head = nn.Sequential(
            OrderedDict([
                ("head_linear_hid", nn.Linear(hidden_dim, hidden_dim)),
                ("head_activation", nn.ReLU(inplace=True)),
                ("head_linear_out", nn.Linear(hidden_dim, output_dim))
            ])
        )
        
        # Akhila # 2 hidden layers in head
        self.two_layer_prediction_head = nn.Sequential(
            OrderedDict([
                ("head_linear_hid1", nn.Linear(hidden_dim, hidden_dim)),
                ("head_activation1", nn.ReLU(inplace=True)),
                ("head_linear_hid2", nn.Linear(hidden_dim, 100)),
                ("head_activation2", nn.ReLU(inplace=True)),
                ("head_linear_out", nn.Linear(100, output_dim))
            ])
        )

    @property
    def encoder(self) -> nn.Module:
        return self.__encoder
        
    def _first_phase_step(self, x: torch.Tensor, x_corrupted: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        emb_anchor = torch.relu(self.encoder(x))
        emb_anchor = self.pretraining_head(emb_anchor)
        emb_anchor = F.normalize(emb_anchor, p=2)
        
        emb_corrupted = torch.relu(self.__encoder(x_corrupted))
        emb_corrupted = self.pretraining_head(emb_corrupted)
        emb_corrupted = F.normalize(emb_corrupted, p=2)

        return emb_anchor, emb_corrupted
    
    def _second_phase_step(self, x) -> torch.Tensor:
        emb = torch.relu(self.encoder(x))
        
        if self.pred_head_size == 1:
            output = self.one_layer_prediction_head(emb)
        else:
            output = self.two_layer_prediction_head(emb)
        
        return output
