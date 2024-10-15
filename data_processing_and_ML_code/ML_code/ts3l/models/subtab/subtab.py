import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Tuple, Union, OrderedDict
from ts3l.models.common import TS3LModule
from ts3l.functional.subtab import arrange_tensors
class ShallowEncoder(nn.Module):
    def __init__(self,
                 feat_dim : int,
                 hidden_dim : int,
                 n_subsets : int,
                 overlap_ratio : float,
                 encoder_depth = 1,
                 dropout_rate = 0.04
    ) -> None:
        super().__init__()

        n_column_subset = int(feat_dim / n_subsets)
        n_overlap = int(overlap_ratio * n_column_subset)

        # Original
        #self.net = nn.Sequential(
        #    nn.Linear(n_column_subset + n_overlap, hidden_dim),
        #    nn.LeakyReLU(),
        #)

        layers = []
        in_dim = n_column_subset + n_overlap
        print(feat_dim, n_subsets, overlap_ratio, n_column_subset)
        print(n_column_subset, n_overlap)
        for _ in range(encoder_depth - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))
        # Combine the layers using nn.Sequential
        self.net = nn.Sequential(*layers)

        
    def forward(self,
                x : torch.Tensor
    ) -> torch.Tensor:
        return self.net(x)

class ShallowDecoder(nn.Module):
    def __init__(self,
                 hidden_dim : int,
                 out_dim : int
    ) -> None:
        super().__init__()

        self.net = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.net(x)

class AutoEncoder(nn.Module):
    def __init__(self,
                 feat_dim : int,
                 hidden_dim : int,
                 n_subsets : int,
                 overlap_ratio : float,
                 encoder_depth = 1,
                 dropout_rate = 0.04
    ) -> None:
        super().__init__()

        self.encoder = ShallowEncoder(feat_dim, hidden_dim, n_subsets, overlap_ratio, encoder_depth, dropout_rate)
        self.decoder = ShallowDecoder(hidden_dim, feat_dim)

        self.projection_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def encode(self, x : torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, x : torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
    
    def forward(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        projection = self.projection_net(latent)
        projection = F.normalize(projection, p = 2, dim = 1)
        x_recon = self.decode(latent)
        return latent, projection, x_recon

class SubTab(TS3LModule):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 n_subsets: int,
                 overlap_ratio: float,
                 encoder_depth = 1,
                 dropout_rate = 0.04
    ) -> None:
        super(SubTab, self).__init__()

        self.feat_dim = input_dim
        
        self.n_subsets = n_subsets
        
        self.__auto_encoder = AutoEncoder(self.feat_dim, hidden_dim, n_subsets, overlap_ratio, encoder_depth, dropout_rate)

        # Original
        #self.head = nn.Sequential(
        #    nn.Linear(hidden_dim, output_dim)
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
        return self.__auto_encoder
    
    def _first_phase_step(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        latents, projections, x_recons = self.__auto_encoder(x)
        
        return projections, x_recons
    
    def _second_phase_step(self, 
                x : torch.Tensor,
                return_embeddings : bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        latent = self.__auto_encoder.encode(x)
        latent = arrange_tensors(latent, self.n_subsets)
        
        latent = latent.reshape(x.shape[0] // self.n_subsets, self.n_subsets, -1).mean(1)

        # Akhila
        if self.pred_head_size == 1:
            out = self.one_layer_prediction_head(latent)
        else:
            out = self.two_layer_prediction_head(latent)
            
        #out = self.head(latent)

        if return_embeddings:
            return out, latent
        return out

