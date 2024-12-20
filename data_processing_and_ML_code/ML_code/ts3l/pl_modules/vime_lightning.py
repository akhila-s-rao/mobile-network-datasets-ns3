from typing import Dict, Any, Tuple
import torch
from torch import nn

from .base_module import TS3LLightining
from ts3l.models import VIME
from ts3l.utils.vime_utils import VIMEConfig
from ts3l import functional as F

class VIMELightning(TS3LLightining):
    
    def __init__(self, config: VIMEConfig) -> None:
        """Initialize the pytorch lightining module of VIME

        Args:
            config (VIMEConfig): The configuration of VIMELightning.
        """
        super(VIMELightning, self).__init__(config)
    
    def _initialize(self, config: Dict[str, Any]) -> None:
        """Initializes the model with specific hyperparameters and sets up various components of VIMELightning.

        Args:
            config (Dict[str, Any]): The given hyperparameter set for VIME. 
        """
        
        self.alpha1 = config["alpha1"]
        self.alpha2 = config["alpha2"]
        del config["alpha1"]
        del config["alpha2"]
        
        self.beta = config["beta"]
        del config["beta"]
        
        self.K = config["K"]
        del config["K"]
        
        self.num_categoricals, self.num_continuous = config["num_categoricals"], config["num_continuous"]
        del config["num_categoricals"]
        del config["num_continuous"]
        
        self.u_label = config["u_label"]
        del config["u_label"]
        
        del config["p_m"]
        
        self.mask_loss_fn = nn.BCELoss()
        # Akhila changed this to fix the not handling output layer of categorical features bug
        #self.categorical_feature_loss_fn = nn.CrossEntropyLoss()
        self.categorical_feature_loss_fn = nn.BCELoss()
        
        self.continuous_feature_loss_fn = nn.MSELoss()
        
        self.consistency_loss_fn = nn.MSELoss()

        self._init_model(VIME, config)
    
    def _get_first_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Calculate the first phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.FloatTensor: The final loss of first phase step
        """
        
        mask_preds, feature_preds = F.vime.first_phase_step(self.model, batch)
        
        mask_loss, categorical_feature_loss, continuous_feature_loss = F.vime.first_phase_loss(
            batch[2][:, : self.num_categoricals],
            batch[2][:, self.num_categoricals :],
            batch[1],
            feature_preds[:, : self.num_categoricals],
            feature_preds[:, self.num_categoricals :],
            mask_preds,
            self.mask_loss_fn,
            self.categorical_feature_loss_fn,
            self.continuous_feature_loss_fn,
        )
        
        return mask_loss + self.alpha1 * categorical_feature_loss + self.alpha2 * continuous_feature_loss
    
    def _get_second_phase_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        """Calculate the second phase loss

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch

        Returns:
            torch.FloatTensor: The final loss of second phase step
            torch.Tensor: The label of the labeled data
            torch.Tensor: The predicted label of the labeled data
        """
        _, y = batch
        #print('shape of X from batch', batch[0].shape)
        
        y_hat = F.vime.second_phase_step(self.model, batch)
        
        labeled_idx = (y != self.u_label).flatten()
        unlabeled_idx = (y == self.u_label).flatten()
        
        labeled_y_hat = y_hat[labeled_idx]
        labeled_y = y[labeled_idx].squeeze()
        
        unlabeled_y_hat = y_hat[unlabeled_idx]
        
        task_loss, consistency_loss = F.vime.second_phase_loss(labeled_y, labeled_y_hat, unlabeled_y_hat, self.consistency_loss_fn, self.task_loss_fn, self.K)
        loss = task_loss + self.beta * consistency_loss
        
        return loss, labeled_y, labeled_y_hat

    # Akhila
    def set_second_phase(self, freeze_encoder: bool = True, pred_head_size: int = 1) -> None:
        """Set the module to fine-tuning
        
        Args:
            freeze_encoder (bool): If True, the encoder will be frozen during fine-tuning. Otherwise, the encoder will be trainable.
                                    Default is True.
        """
        # Akhila
        return super().set_second_phase(freeze_encoder, pred_head_size)
    
    def predict_step(self, batch, batch_idx: int
        ) -> torch.FloatTensor:
        """The predict step of VIME

        Args:
            batch (Dict[str, Any]): The input batch
            batch_idx (int): Only for compatibility, do not use

        Returns:
            torch.FloatTensor: The predicted output (logit)
        """
        y_hat = self(batch[0])

        return y_hat